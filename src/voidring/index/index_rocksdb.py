from typing import Type, Any, Optional, Dict, List, get_origin, Union, Iterator, Set, Tuple
from rocksdict import Options, Rdict, WriteBatch, DBCompressionType

from ..base_rocksdb import BaseRocksDB
from .accessor import AccessorRegistry
from .path_parser import PathParser
from datetime import datetime

import hashlib
import base64
import re

class CollectionInfo:
    """集合信息类"""
    def __init__(self, name: str, model_class: Type, cf_name: str = None):
        self.name = name
        self.model_class = model_class
        self.cf_name = cf_name
        self.field_paths = set()  # 索引的字段路径
        
    def add_field_path(self, field_path: str):
        self.field_paths.add(field_path)
        
    def __repr__(self):
        return f"CollectionInfo(name='{self.name}', model={self.model_class.__name__}, indexes={list(self.field_paths)})"

class IndexedRocksDB(BaseRocksDB):
    """
    支持针对一个集合的多种索引路径管理。

    在增加、删除、修改对象时根据注册的索引路径自动更新索引。
    """

    def __init__(self, path: str = None, *args, **kwargs):
        super().__init__(path, *args, **kwargs)

        # 创建索引元数据的列族
        all_cfs = self.list_column_families(self.path)
        if self.INDEX_METADATA_CF not in all_cfs:
            self.create_column_family(self.INDEX_METADATA_CF, options=self._get_indexes_cf_options())
        if self.INDEX_CF not in all_cfs:
            self.create_column_family(self.INDEX_CF, options=self._get_indexes_cf_options())
            
        # 内存中保存集合信息的字典，而不是存储在RocksDB中
        self._collections: Dict[str, CollectionInfo] = {}

    INDEX_METADATA_CF = "indexes_metadata"  # 索引元数据列族
    INDEX_CF = "indexes"      # 索引列族

    # 集合前缀
    COLLECTION_PREFIX_FORMAT = "idx:{cf_name}:{collection_name}"

    # 索引元数据格式
    INDEX_METADATA_FORMAT = COLLECTION_PREFIX_FORMAT + ":{field_path}"

    # 索引格式
    INDEX_KEY_FORMAT = INDEX_METADATA_FORMAT + ":{value}:key:{key}"

    # 关键标识符
    RESERVED_WORD_IN_INDEX = ":key:"

    # 特殊字符替换映射
    SPECIAL_CHARS = {
        '.': '_dot_',
        '[': '_lb_',
        ']': '_rb_',
        '{': '_lcb_',
        '}': '_rcb_',
        ':': '_col_',
        '/': '_sl_',
        '\\': '_bs_',
        '*': '_ast_',
        '?': '_qm_',
        '<': '_lt_',
        '>': '_gt_',
        '|': '_pipe_',
        '"': '_quot_',
        "'": '_apos_'
    }
    
    _accessor_registry = AccessorRegistry()
    _path_parser = PathParser()

    @classmethod
    def _get_indexes_cf_options(cls) -> Options:
        """
        获取专门为索引列族优化的rocksdb列族配置
        """
        options = Options()
        options.set_write_buffer_size(64 * 1024 * 1024)  # 64MB
        options.set_max_write_buffer_number(4)  # 允许更多的写缓冲
        options.set_min_write_buffer_number_to_merge(1)  # 尽快刷新到L0
        options.set_target_file_size_base(64 * 1024 * 1024)  # 64MB
        options.set_compression_type(DBCompressionType.none())  # 禁用压缩，因为索引值都是None
        options.set_bloom_locality(1)  # 优化布隆过滤器的局部性
        return options

    @classmethod
    def _get_base_type(cls, model_class: Type) -> Type:
        """获取基础类型"""
        
        # 处理 typing 类型
        if hasattr(model_class, '__origin__'):
            return model_class.__origin__
            
        # 处理内置类型
        if model_class in (dict, Dict):
            return dict
            
        # 其他类型保持不变
        return model_class

    @classmethod
    def _escape_special_chars(cls, value: str) -> str:
        """替换字符串中的特殊字符，使用Base64编码
        
        为了提高性能:
        1. 使用预编译的替换映射表
        2. 只在字符串包含特殊字符时才进行替换
        3. 使用 translate() 方法进行批量替换
        """
        # 使用类变量缓存编码映射表,避免重复计算
        if not hasattr(cls, '_ENCODE_MAP'):
            cls._ENCODE_MAP = str.maketrans({
                char: f"_B64_{base64.b64encode(char.encode()).decode()}_" 
                for char in cls.SPECIAL_CHARS
            })
            
        # 快速检查是否需要编码
        if not any(c in value for c in cls.SPECIAL_CHARS):
            return value
            
        return value.translate(cls._ENCODE_MAP)

    @classmethod
    def _unescape_special_chars(cls, value: str) -> str:
        """还原被Base64编码的特殊字符
        
        为了提高性能:
        1. 使用预编译的还原映射表
        2. 只在字符串包含编码标记时才进行还原
        3. 使用正则表达式一次性匹配所有编码
        """
        # 使用类变量缓存解码映射表
        if not hasattr(cls, '_DECODE_MAP'):
            cls._DECODE_MAP = {
                f"_B64_{base64.b64encode(char.encode()).decode()}_": char
                for char in cls.SPECIAL_CHARS
            }
            # 预编译正则表达式
            cls._B64_PATTERN = re.compile('|'.join(map(re.escape, cls._DECODE_MAP.keys())))
            
        # 快速检查是否需要解码
        if '_B64_' not in value:
            return value
            
        return cls._B64_PATTERN.sub(lambda m: cls._DECODE_MAP[m.group()], value)

    @classmethod
    def _fetch_key_from_index(cls, index_key: str) -> str:
        """从索引中获取键"""
        parts = index_key.rsplit(cls.RESERVED_WORD_IN_INDEX, 1)
        if len(parts) != 2:
            raise ValueError(f"从索引键 {index_key} 中提取键失败")
        return cls._unescape_special_chars(parts[1])

    @classmethod
    def _fetch_field_path_from_index(cls, index_key: str) -> str:
        """从索引中获取字段路径"""
        parts = index_key.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"从索引键 {index_key} 中提取字段路径失败")
        return parts[1]

    @classmethod
    def format_index_value(cls, value: Any) -> str:
        """格式化索引值
        
        格式化规则：
        1. None -> "null"
        2. 布尔值 -> "false" 或 "true"
        3. 数值：
            - float('-inf') -> "a" (确保小于所有数值)
            - 负数 -> "a{数值}" (确保小于正数)
            - 0 -> "c0000000000_000000"
            - 正数 -> "c{数值}"
            - float('inf') -> "d" (确保大于所有数值)
            - float('nan') -> "e" (确保排在最后)
        4. 日期时间 -> "t{timestamp:010d}"
        5. 字符串：
            - 空字符串 -> "empty"
            - 长字符串 -> base32编码的MD5哈希
            - 普通字符串 -> 转义后的字符串
        """
        if value is None:
            return 'null'
            
        if isinstance(value, bool):
            return str(value).lower()  # 使用小写以确保排序一致性
            
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                if value == float('inf'): return 'd'
                if value == float('-inf'): return 'a'
                if value != value: return 'e'
            
            num = float(value)
            if num == 0:
                return 'c0000000000_000000'
            
            abs_num = abs(num)
            int_part = int(abs_num)
            dec_part = int((abs_num - int_part) * 1e6)
            
            if num < 0:
                # 负数：按位对齐做减法
                int_part_str = f"{9999999999 - int_part:010d}"
                dec_part_str = f"{999999 - dec_part:06d}"
                result = f"b{int_part_str}_{dec_part_str}"
            else:
                # 正数：直接格式化
                result = f"c{int_part:010d}_{dec_part:06d}"
            
            return result
            
        if isinstance(value, datetime):
            return f"t{int(value.timestamp()):010d}"
            
        if isinstance(value, str):
            if not value:
                return 'empty'
            if len(value) > 100:
                import base64
                hash_bytes = hashlib.md5(value.encode()).digest()
                return f"h{base64.b32encode(hash_bytes).decode().rstrip('=')}"
            # 添加前缀 's' 以区分字符串类型
            return f"s{cls._escape_special_chars(value)}"
            
        # 其他类型转为字符串
        return f"v{cls._escape_special_chars(str(value))}"

    @property
    def indexes_metadata_cf(self) -> Rdict:
        return self.get_column_family(self.INDEX_METADATA_CF)

    @property
    def indexes_cf(self) -> Rdict:
        return self.get_column_family(self.INDEX_CF)

    def validate_path(self, model_class: Type, field_path: str) -> None:
        """验证字段路径是否可以访问到属性值"""
        if field_path != "#":
            self._accessor_registry.validate_path(model_class, field_path)
    
    def get_field_value(self, obj: Any, field_path: str, key: str) -> Any:
        """获取字段值，支持字典和对象"""
        if field_path == "#":
            return key
            
        # 处理字典类型 - 计算属性特殊处理
        if isinstance(obj, dict) and field_path in self._collections.get(obj.get("_collection", ""), CollectionInfo("", dict)).field_paths:
            # 尝试将字典转换回模型实例以获取计算属性
            try:
                collection_info = self._collections.get(obj.get("_collection", ""))
                if collection_info and hasattr(collection_info.model_class, field_path):
                    # 检查是否为属性
                    if isinstance(getattr(collection_info.model_class, field_path, None), property):
                        # 临时将字典转换回模型实例
                        instance = collection_info.model_class.model_validate(obj)
                        return getattr(instance, field_path)
            except Exception as e:
                self._logger.debug(f"计算属性获取失败: {e}")
            
            # 常规字典值获取
            parts = field_path.split(".")
            value = obj
            for part in parts:
                if part in value:
                    value = value[part]
                else:
                    return None
            return value
        else:
            # 原有逻辑处理对象
            return self._accessor_registry.get_field_value(obj, field_path)
    
    def register_collections(self, collection_name: str, model_class: Type, field_paths: List[str], cf_name: str=None):
        """批量注册集合的索引配置"""
        for field_path in field_paths:
            self.register_index(collection_name, model_class, field_path, cf_name)
            
    def get_collections(self) -> List[CollectionInfo]:
        """获取所有注册的集合信息"""
        return list(self._collections.values())
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """获取指定集合的信息"""
        return self._collections.get(collection_name)

    def register_collection(self, collection_name: str, model_class: Type, cf_name: str=None):
        """
        注册集合

        如果要使用 iter_collection_keys 或 rebuild_indexes 方法，必须先注册集合。
        """
        cf_name = cf_name or self.default_cf_name
        
        # 在内存中注册集合
        if collection_name not in self._collections:
            self._collections[collection_name] = CollectionInfo(collection_name, model_class, cf_name)
            
        # 注册主键索引
        self.register_index(collection_name, model_class, field_path="#", cf_name=cf_name)

    def register_index(self, collection_name: str, model_class: Type, field_path: str, cf_name: str=None):
        """注册集合的索引配置"""

        # 验证字段路径的语法是否合法
        try:
            if field_path != "#":
                self._path_parser.parse(field_path)
        except ValueError as e:
            self._logger.error(f"字段路径 '{field_path}' 格式无效: {str(e)}")
            raise ValueError(f"无效的字段路径 '{field_path}': {str(e)}")

        # 验证字段路径是否可以访问到属性值
        try:
            self.validate_path(model_class, field_path)
        except Exception as e:
            self._logger.error(f"字段路径验证失败: {e}")
            raise

        # 确保集合已注册
        cf_name = cf_name or self.default_cf_name
        if collection_name not in self._collections:
            self._collections[collection_name] = CollectionInfo(collection_name, model_class, cf_name)
        
        # 添加字段路径到集合信息
        self._collections[collection_name].add_field_path(field_path)

        # 构建索引键
        key = self.INDEX_METADATA_FORMAT.format(
            cf_name=cf_name,
            collection_name=collection_name,
            field_path=field_path
        )

        # 内存中已有元数据，仅在RocksDB创建索引标识
        try:
            existing_entry = self.indexes_metadata_cf.get(key)
            if existing_entry is None:  # 确实不存在
                # 只存储一个标记，不再存储类型
                self.indexes_metadata_cf[key] = True
                self._logger.debug(f"注册索引元数据: {key} -> {cf_name}.{collection_name}#{field_path}")
            else:
                self._logger.debug(f"索引元数据已存在: {key}")
        except KeyError:  # 确实不存在
            self.indexes_metadata_cf[key] = True
            self._logger.debug(f"注册索引元数据: {key} -> {cf_name}.{collection_name}#{field_path}")

    def _make_index_key(self, collection_name: str, field_path: str, field_value: Any, key: str, cf_name: str=None) -> str:
        """创建索引键"""
        key = self._escape_special_chars(key)
        formatted_value = self.format_index_value(field_value)
        cf_name = cf_name or self.default_cf_name
        return self.INDEX_KEY_FORMAT.format(
            cf_name=cf_name,
            collection_name=collection_name,
            field_path=field_path,
            value=formatted_value,
            key=key
        )
    
    def update_with_indexes(self, collection_name: str, key: str, value: Any, cf_name: str=None):
        """更新键值，并自动更新索引"""
        
        # 自动检测并转换Pydantic模型
        if hasattr(value, "model_dump"):  # Pydantic v2
            value = value.model_dump()
            value["_collection"] = collection_name  # 添加集合标记
        elif hasattr(value, "dict"):      # Pydantic v1
            value = value.dict()
        
        self._logger.debug(f"开始更新索引: collection={collection_name}, key={key}, value=`{str(value)[:300]}`, cf_name={cf_name}")

        cf_name = cf_name or self.default_cf_name
        cf = self.get_column_family(cf_name)
        cf_handle = self.get_column_family_handle(cf_name)

        key_existing, old_value = self.key_exist(key, rdict=cf)

        # 获取对象所有路径
        collection_prefix = self.COLLECTION_PREFIX_FORMAT.format(cf_name=cf_name, collection_name=collection_name)

        all_paths = self.keys(prefix=collection_prefix, rdict=self.indexes_metadata_cf)

        if not all_paths:
            cf.put(key, value)
            self._logger.debug(f"值已更新，但没有索引注册")
            return

        # 处理删除操作
        batch = WriteBatch()

        # 更新值
        batch.put(key, value, cf_handle)

        # 处理对象所有属性访问路径的索引
        indexes_cf_handle = self.get_column_family_handle(self.INDEX_CF)
        for path in all_paths:
            field_path = self._fetch_field_path_from_index(path)
            if key_existing:
                field_value = self.get_field_value(old_value, field_path, key)
                old_index = self._make_index_key(
                    collection_name=collection_name,
                    field_path=field_path,
                    field_value=field_value,
                    key=key,
                    cf_name=cf_name
                )
                batch.delete(old_index, indexes_cf_handle)
                self._logger.debug(f"准备删除旧索引: {old_index}")
            field_value = self.get_field_value(value, field_path, key)
            new_index = self._make_index_key(
                collection_name=collection_name,
                field_path=field_path,
                field_value=field_value,
                key=key,
                cf_name=cf_name
            )
            batch.put(new_index, None, indexes_cf_handle)
            self._logger.debug(f"准备创建新索引: {new_index}")

        self.write(batch)
        self._logger.debug(f"批处理任务提交完成，值和索引已更新")

    def delete_with_indexes(self, collection_name: str, key: str, cf_name: str=None):
        """删除键值，并自动删除索引"""
        self._logger.debug(f"开始删除索引: collection={collection_name}, key={key}, cf_name={cf_name}")

        cf_name = cf_name or self.default_cf_name
        cf = self.get_column_family(cf_name)

        key_existing, old_value = self.key_exist(key, rdict=cf)
        if not key_existing:
            self._logger.debug(f"不存在旧值，无需删除")
            return

        # 处理删除操作
        batch = WriteBatch()

        # 更新值
        cf_handle = self.get_column_family_handle(cf_name)
        batch.delete(key, cf_handle)

        # 获取对象所有路径
        collection_prefix = self.COLLECTION_PREFIX_FORMAT.format(cf_name=cf_name, collection_name=collection_name)

        all_paths = self.keys(prefix=collection_prefix, rdict=self.indexes_metadata_cf)

        if not all_paths:
            self.write(batch)
            self._logger.debug(f"批处理任务提交完成，值已删除，但没有索引注册")
            return

        # 处理对象所有属性访问路径的索引
        indexes_cf_handle = self.get_column_family_handle(self.INDEX_CF)
        for path in all_paths:
            field_path = self._fetch_field_path_from_index(path)
            field_value = self.get_field_value(old_value, field_path, key)
            old_index = self._make_index_key(
                collection_name=collection_name,
                field_path=field_path,
                field_value=field_value,
                key=key,
                cf_name=cf_name
            )
            batch.delete(old_index, indexes_cf_handle)
            self._logger.debug(f"准备删除索引: {old_index}")

        self.write(batch)
        self._logger.debug(f"批处理任务提交完成，值和索引已删除")        

    def iter_keys_with_index(
        self,
        collection_name: str,
        field_path: str,
        field_value: Any = None, 
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        limit: Optional[int] = None,
        reverse: bool = False,
        cf_name: str = None,
    ) -> Iterator[str]:
        """通过索引查询键"""
        index_cf = self.get_column_family(self.INDEX_CF)
        
        # 构建基础前缀
        cf_name = cf_name or self.default_cf_name
        collection_prefix = self.COLLECTION_PREFIX_FORMAT.format(cf_name=cf_name, collection_name=collection_name)
        
        if start is not None or end is not None:
            # 范围查询时，确保只查询指定字段的索引
            start_key = self._make_index_key(
                collection_name=collection_name,
                field_path=field_path,
                field_value=start,
                key="",
                cf_name=cf_name
            )
            end_key = self._make_index_key(
                collection_name=collection_name,
                field_path=field_path,
                field_value=end,
                key="",
                cf_name=cf_name
            )
            target_key = None
        else:
            start_key = None
            end_key = None
            target_key = self._make_index_key(
                collection_name=collection_name,
                field_path=field_path,
                field_value=field_value,
                key="",
                cf_name=cf_name
            )

        self._logger.debug(f"范围查询: start={start_key}, end={end_key}")
        resp = self.iter_keys(
            prefix=target_key,
            start=start_key,
            end=end_key,
            limit=limit,
            reverse=reverse,
            rdict=index_cf
        )
        for index in resp:
            key = self._fetch_key_from_index(index)
            yield key
    
    def iter_items_with_index(self, collection_name: str, *args, model_class=None, return_as_model=True, **kwargs):
        """迭代索引匹配的键值对
        
        Args:
            collection_name: 集合名称
            *args: 传递给iter_keys_with_index的位置参数
            model_class: 要转换成的Pydantic模型类，如果为None但return_as_model为True则尝试使用注册的模型
            return_as_model: 是否将值转换为模型实例，默认为True
            **kwargs: 传递给iter_keys_with_index的关键字参数
        """
        # 如果没有指定model_class但需要转换，尝试从注册表中获取
        if model_class is None and return_as_model and collection_name in self._collections:
            model_class = self._collections[collection_name].model_class
            
        for key in self.iter_keys_with_index(collection_name, *args, **kwargs):
            value = self.get(key, rdict=kwargs.get("rdict", None))
            
            if return_as_model and model_class and isinstance(value, dict):
                try:
                    value = model_class.model_validate(value)
                except Exception as e:
                    self._logger.warning(f"转换模型失败: {e}")
                    # 转换失败时仍返回原始值
            
            yield key, value

    def items_with_index(self, *args, **kwargs):
        return list(self.iter_items_with_index(*args, **kwargs))

    def keys_with_index(self, *args, **kwargs):
        return [k for k, _ in self.iter_items_with_index(*args, **kwargs)]

    def values_with_index(self, *args, **kwargs):
        return [v for _, v in self.iter_items_with_index(*args, **kwargs)]

    def iter_collection_keys(self, collection_name: str, cf_name: str=None):
        """迭代集合的所有键，即查找集合的所有实例。"""
        cf_name = cf_name or self.default_cf_name
        collection_keys_prefix = self.COLLECTION_PREFIX_FORMAT.format(cf_name=cf_name, collection_name=collection_name) + ":#:"
        resp = self.iter_keys(prefix=collection_keys_prefix, rdict=self.indexes_cf)
        self._logger.debug(f"iter_collection_keys: {resp}")
        for index in resp:
            key = self._fetch_key_from_index(index)
            self._logger.debug(f"iter_collection_keys: {key}")
            yield key
            
    def iter_collection(self, collection_name: str, cf_name: str=None, return_as_model=True):
        """迭代集合中的所有项目"""
        cf_name = cf_name or self.default_cf_name
        cf = self.get_column_family(cf_name)
        
        # 获取模型类
        model_class = None
        if return_as_model and collection_name in self._collections:
            model_class = self._collections[collection_name].model_class
            
        for key in self.iter_collection_keys(collection_name, cf_name):
            value = cf.get(key)
            
            if return_as_model and model_class and isinstance(value, dict):
                try:
                    value = model_class.model_validate(value)
                except Exception as e:
                    self._logger.warning(f"转换模型失败: {e}")
            
            yield key, value

    def rebuild_indexes(self, collection_name: str, cf_name: str=None):
        """重建集合所有实例的索引"""
        cf_name = cf_name or self.default_cf_name
        cf = self.get_column_family(cf_name)

        for key in self.iter_collection_keys(collection_name, cf_name):
            self.update_with_indexes(
                collection_name=collection_name,
                key=key,
                value=cf[key],
                cf_name=cf_name
            )
        self._logger.debug(f"重建索引完成")

    def get_as_model(self, collection_name: str, key: str, model_class=None, cf_name=None):
        """获取数据并自动转换为模型实例"""
        # 如果未指定model_class，尝试从注册表获取
        if model_class is None and collection_name in self._collections:
            model_class = self._collections[collection_name].model_class
            
        if model_class is None:
            self._logger.warning(f"无法找到集合 {collection_name} 的模型类定义")
            return self.get(key, rdict=self.get_column_family(cf_name or self.default_cf_name))
            
        data = self.get(key, rdict=self.get_column_family(cf_name or self.default_cf_name))
        if data and isinstance(data, dict):
            try:
                return model_class.model_validate(data)
            except Exception as e:
                self._logger.warning(f"模型转换失败: {e}")
                return data
        return data