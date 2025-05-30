import logging
import itertools
import os

from rocksdict import Rdict, Options, WriteBatch, SstFileWriter, ReadOptions, WriteOptions
from typing import Any, Iterator, Optional, Union, Tuple, Literal
from dataclasses import dataclass
from enum import Enum
from itertools import islice

class BaseRocksDB:
    """RocksDB基础封装类
    
    提供核心的键值存储功能，支持：
    1. 基础的键值操作
    2. 可选的Rdict实例（批处理、列族等）
    3. 默认列族访问
    
    Examples:
        # 基础使用
        with BaseRocksDB("path/to/db") as db:
            db["key"] = value
            
        # 使用列族
        with BaseRocksDB("path/to/db") as db:
            default_cf = db.default_cf
            users_cf = db.get_column_family("users")
            
            # 写入不同列族
            db.put("key", value, default_cf)  # 等同于 db["key"] = value
            db.put("user:1", user_data, users_cf)
            
        # 批量写入
        with BaseRocksDB("path/to/db") as db:
            with db.batch_write() as batch:
                db.put("key1", value1, batch)
                db.put("key2", value2, batch)
    """

    def __init__(
        self,
        path: str,
        *args,
        **kwargs
    ):
        """初始化BaseRocksDB
        
        Args:
            path: 数据库路径
            options: 可选的RocksDB配置
            logger: 可选的日志记录器
        """
        self.path = path
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据库路径不存在: {self.path}")

        self._db = Rdict(self.path, *args, **kwargs)
        self._logger = logging.getLogger(__name__)
        self._default_cf = self._db.get_column_family("default")
        self._default_cf_name = "default"

    def key_exist(
        self,
        key: Any,
        *,
        rdict: Optional[Rdict] = None,
        options: Optional[ReadOptions] = None,
    ) -> Tuple[bool, Optional[Any]]:
        """快速检查键是否存在
        
        Returns:
            (exists, value):
                - (True, value): 键确定存在且返回值
                - (False, None): 键可能存在但需要进一步确认
        """
        target = rdict if rdict is not None else self._db
        existing, value = target.key_may_exist(key, True, options)
        if existing and value is not None:
            # 布隆过滤器说找到了
            return True, value
        elif not existing:
            return False, None
        else:
            # 可能存在或不存在，尝试直接获取
            try:
                value = target.get(key, options)
                self._logger.debug(f"may_exist: {key} -> {value}")
                return True, value if value is not None else None

            except KeyError:
                return False, None

    def put(
        self,
        key: Any,
        value: Any,
        *,
        rdict: Optional[Rdict] = None,
        options: Optional[WriteOptions] = None,
    ) -> None:
        """写入数据
        
        Args:
            key: 数据键
            value: 要写入的值
            rdict: 可选的Rdict实例（如批处理器、列族等）
            options: 写入选项
            
        Examples:
            # 基本写入
            db.put("key", "value")
            
            # 使用写入选项
            opts = WriteOptions()
            opts.disable_wal(True)  # 禁用预写日志以提高性能
            db.put("key", "value", options=opts)
            
            # 写入列族
            users_cf = db.get_column_family("users")
            db.put("user:1", user_data, rdict=users_cf)
        """
        target = rdict if rdict is not None else self._db
        target.put(key, value, options)
        self._logger.debug(f"put: {key} -> {value}")

    def delete(self, key: Any, rdict: Optional[Rdict] = None) -> None:
        """删除数据
        
        Args:
            key: 要删除的键
            rdict: 可选的Rdict实例（如批处理器、列族等）
        """
        target = rdict if rdict is not None else self._db
        del target[key]
        self._logger.debug(f"delete: {key}")

    def get(
        self,
        key: Union[Any, list[Any]],
        *,
        default: Any = None,
        rdict: Optional[Rdict] = None,
        options: Optional[ReadOptions] = None,
    ) -> Any:
        """获取数据
        
        Args:
            key: 单个键或键列表
            default: 键不存在时的默认返回值
            rdict: 可选的Rdict实例（如列族等）
            options: 读取选项
            
        Returns:
            存储的值，如果键不存在则返回默认值
        """
        target = rdict if rdict is not None else self._db
        try:
            return target.get(key, default, options)
        except KeyError:
            return default

    def iter(
        self,
        *,
        rdict: Optional[Rdict] = None,
        prefix: Optional[str] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        reverse: bool = False,
        fill_cache: bool = True,
        options: Optional[ReadOptions] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        """返回键值对迭代器
        
        Args:
            rdict: 可选的RocksDict实例
            prefix: 键前缀
            start: 起始键（包含）
            end: 结束键（不包含）
            reverse: 是否反向迭代
            fill_cache: 是否填充缓存
            options: 读取选项
        """
        target = rdict if rdict is not None else self._db
        
        opts = options or ReadOptions()
        if not fill_cache:
            opts.fill_cache(False)
        
        it = target.iter(opts)
        
        # 处理前缀搜索的边界
        if prefix is not None and prefix != "":
            if start is None:
                start = prefix
            if end is None:
                # 创建一个比前缀大的最小字符串作为上界
                end = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        
        # 如果是反向迭代且 start > end，交换它们
        if reverse and start is not None and end is not None and start > end:
            start, end = end, start
        
        # 设置迭代器起始位置
        if reverse:
            if end is not None:
                it.seek_for_prev(end)
            else:
                it.seek_to_last()
        else:
            if start is not None:
                it.seek(start)
            else:
                it.seek_to_first()
        
        # 检查迭代器是否有效
        if not it.valid():
            if reverse and start is not None:
                it.seek_for_prev(start)
            return
        
        # 迭代并应用过滤
        while it.valid():
            key = it.key()
            
            # 检查范围
            if reverse:
                if start is not None and key < start:
                    break
                if end is not None and key >= end:
                    it.prev()
                    continue
            else:
                if end is not None and key >= end:
                    break
                if start is not None and key < start:
                    it.next()
                    continue
            
            # 检查前缀
            if prefix is not None and prefix != "":
                if not key.startswith(prefix):
                    if reverse:
                        if key < prefix:
                            break
                        it.prev()
                        continue
                    else:
                        if key > prefix:
                            break
                        it.next()
                        continue
            
            try:
                yield key, it.value()
            except Exception as e:
                self._logger.error(f"iter error: {e}")
                break
            
            if reverse:
                it.prev()
            else:
                it.next()

    def items(
        self,
        *,
        rdict: Optional[Rdict] = None,
        prefix: Optional[str] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        reverse: bool = False,
        limit: Optional[int] = None,
        fill_cache: bool = True,
        options: Optional[ReadOptions] = None,
    ) -> list[Tuple[Any, Any]]:
        """返回键值对列表
        
        Args:
            options: 自定义读取选项
            rdict: 可选的Rdict实例（如列族等）
            prefix: 键前缀过滤
            start: 起始键（包含）
            end: 结束键（不包含）
            reverse: 是否反向迭代
            limit: 限制返回的项目数量
            fill_cache: 是否将扫描的数据填充到块缓存中
        """
        iterator = self.iter(
            rdict=rdict,
            prefix=prefix,
            start=start,
            end=end,
            reverse=reverse,
            fill_cache=fill_cache,
            options=options,
        )
        if limit is not None:
            return list(islice(iterator, limit))
        return list(iterator)
    
    def keys(self, *args, limit: Optional[int] = None, **kwargs) -> list[Any]:
        """返回键列表
        
        Args:
            *args: 传递给 items 的位置参数
            limit: 限制返回的键数量
            **kwargs: 传递给 items 的关键字参数
        """
        iterator = (k for k, _ in self.iter(*args, **kwargs))
        if limit is not None:
            return list(islice(iterator, limit))
        return list(iterator)
    
    def values(self, *args, limit: Optional[int] = None, **kwargs) -> list[Any]:
        """返回值列表
        
        Args:
            *args: 传递给 items 的位置参数
            limit: 限制返回的值数量
            **kwargs: 传递给 items 的关键字参数
        """
        iterator = (v for _, v in self.iter(*args, **kwargs))
        if limit is not None:
            return list(islice(iterator, limit))
        return list(iterator)
    
    def iter_keys(self, *args, limit: Optional[int] = None, **kwargs) -> Iterator[Any]:
        """返回键迭代器
        
        Args:
            *args: 传递给 iter 的位置参数
            limit: 限制返回的键数量
            **kwargs: 传递给 iter 的关键字参数
        """
        iterator = (k for k, _ in self.iter(*args, **kwargs))
        if limit is not None:
            yield from islice(iterator, limit)
        else:
            yield from iterator
    
    def iter_values(self, *args, limit: Optional[int] = None, **kwargs) -> Iterator[Any]:
        """返回值迭代器
        
        Args:
            *args: 传递给 iter 的位置参数
            limit: 限制返回的值数量
            **kwargs: 传递给 iter 的关键字参数
        """
        iterator = (v for _, v in self.iter(*args, **kwargs))
        if limit is not None:
            yield from islice(iterator, limit)
        else:
            yield from iterator
    
    def write(self, batch: WriteBatch) -> None:
        """执行批处理
        
        Args:
            batch: 要执行的批处理实例
            
        Examples:
            batch = db.batch()
            try:
                batch.put(key1, value1)
                batch.put(key2, value2)
                db.write(batch)
            except Exception as e:
                logger.error(f"Batch operation failed: {e}")
                raise
        """
        self._logger.debug(f"write with batch: {batch.len()} items")
        self._db.write(batch)
    
    def close(self) -> None:
        """关闭数据库"""
        self._db.close()
    
    @classmethod
    def destroy(cls, path: str, options: Optional[Options] = None) -> None:
        """删除数据库"""
        options = options or Options()
        Rdict.destroy(path, options) 

    @property
    def default_cf(self):
        """获取默认列族"""
        return self._default_cf
    
    @property
    def default_cf_name(self):
        """获取默认列族名称"""
        return self._default_cf_name
    
    def get_column_family(self, name: str) -> Rdict:
        """获取指定名称的列族"""
        return self._db.get_column_family(name)

    @classmethod
    def list_column_families(cls, path: str, options: Optional[Options] = None) -> list[str]:
        """列举数据库中的所有列族
        
        Args:
            path: 数据库路径
            options: 可选的配置项
            
        Returns:
            列族名称列表
        """
        options = options or Options()
        return Rdict.list_cf(path, options)
    
    def create_column_family(self, name: str, options: Optional[Options] = None) -> Rdict:
        """创建新的列族
        
        Args:
            name: 列族名称
            options: 可选的列族配置
            
        Returns:
            新创建的列族实例
        """
        options = options or Options()
        cf = self._db.create_column_family(name, options)
        self._logger.debug(f"create_column_family: {cf}")
        return cf
    
    def drop_column_family(self, name: str) -> None:
        """删除指定的列族
        
        Args:
            name: 要删除的列族名称
        """
        self._db.drop_column_family(name)
        self._logger.debug(f"drop_column_family: {name}")
    
    def get_column_family_handle(self, name: str):
        """获取列族句柄（用于批处理操作）
        
        Args:
            name: 列族名称
            
        Returns:
            列族句柄
            
        Examples:
            with db.batch_write() as batch:
                cf_handle = db.get_column_family_handle("users")
                batch.put(key, value, cf_handle)
        """
        return self._db.get_column_family_handle(name) 

    def __getitem__(self, key: Any) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)
    
    def __delitem__(self, key: Any) -> None:
        self.delete(key)

    def paginate(
        self, 
        *,
        page_size: int = 10,
        cursor: Optional[str] = None,
        include_cursor: bool = True,
        **kwargs
    ) -> dict:
        """执行基于游标的分页查询
        
        Args:
            page_size: 每页返回的最大项目数
            cursor: 上一页返回的游标，首页不传
            include_cursor: 是否在结果中包含当前游标
            **kwargs: 传递给items方法的其他参数（如prefix, reverse, rdict等）
            
        Returns:
            dict: 包含以下字段:
                items: 当前页数据列表 [(key, value), ...]
                has_more: 是否还有更多数据
                next_cursor: 下一页游标，最后一页为None
                prev_cursor: 上一页游标，首页为None
        """
        # 处理游标
        if cursor:
            # 解码游标并设置起始位置
            kwargs['start'] = self._decode_cursor(cursor)
        
        # 确保不会与自己定义的limit冲突
        if 'limit' in kwargs:
            del kwargs['limit']
        
        # 执行查询
        items = self.items(limit=page_size + 1, **kwargs)
        
        # 判断是否有下一页
        has_more = len(items) > page_size
        if has_more:
            items = items[:page_size]  # 移除额外获取的项
        
        # 计算下一页游标
        next_cursor = None
        if has_more and items:
            next_cursor = self._encode_cursor(items[-1][0])
        
        # 构建结果
        result = {
            'items': items,
            'has_more': has_more,
            'next_cursor': next_cursor
        }
        
        if include_cursor and cursor:
            result['prev_cursor'] = cursor
        
        return result

    def _encode_cursor(self, key):
        """将键编码为游标"""
        import base64
        return base64.urlsafe_b64encode(str(key).encode()).decode()

    def _decode_cursor(self, cursor):
        """从游标解码键"""
        import base64
        key = base64.urlsafe_b64decode(cursor.encode()).decode()
        # 获取比这个key稍大的值作为起始点（排除当前key）
        return key + '\0'



