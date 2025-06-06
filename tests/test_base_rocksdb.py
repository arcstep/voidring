import pytest
from rocksdict import Rdict, Options, ReadOptions, WriteOptions
from voidring.base_rocksdb import BaseRocksDB, WriteBatch
import tempfile
import shutil
import itertools

# 配置日志
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestBasicOperations:
    @pytest.fixture(autouse=True)
    def setup_logging(self, caplog):
        caplog.set_level(logging.INFO)

    @pytest.fixture
    def db_path(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)
    
    @pytest.fixture
    def db(self, db_path):
        db = BaseRocksDB(db_path)
        try:
            yield db
        finally:
            db.close()
            
    def test_put_get_delete(self, db):
        """测试基本的读写删除操作"""
        # 直接方法调用
        db.put("key1", "value1")
        assert db.get("key1") == "value1"
        
        # 字典风格访问
        db["key2"] = "value2"
        assert db["key2"] == "value2"
        
        # 删除操作
        db.delete("key1")
        assert db.get("key1") is None
        del db["key2"]
        assert db["key2"] is None
        
        # 默认值
        assert db.get("nonexistent", default="default") == "default"
        
        # 批量获取
        db.put("multi1", "val1")
        db.put("multi2", "val2")
        results = db.get(["multi1", "multi2", "nonexistent"])
        assert results == ["val1", "val2", None]
    
    def test_collection_methods(self, db):
        """测试集合类方法"""
        # 准备测试数据
        test_data = {
            "user:1": "alice",
            "user:2": "bob",
            "config:1": "setting1"
        }
        for k, v in test_data.items():
            db[k] = v
            
        # 测试 keys()
        all_keys = db.keys()
        assert len(all_keys) == 3
        assert "user:1" in all_keys
        
        user_keys = db.keys(prefix="user:")
        assert len(user_keys) == 2
        assert all(k.startswith("user:") for k in user_keys)
        
        # 测试 values()
        all_values = db.values()
        assert len(all_values) == 3
        assert "alice" in all_values
        
        user_values = db.values(prefix="user:")
        assert len(user_values) == 2
        assert "alice" in user_values
        assert "setting1" not in user_values
        
        # 测试 items()
        all_items = db.items()
        assert len(all_items) == 3
        assert ("user:1", "alice") in all_items
        
        # 测试带限制的 items()
        limited_items = db.items(limit=2)
        assert len(limited_items) == 2
        
        # 测试迭代器方法
        key_count = sum(1 for _ in db.iter_keys())
        assert key_count == 3
        
        value_count = sum(1 for _ in db.iter_values())
        assert value_count == 3
    
    def test_existence_checks(self, db):
        """测试存在性检查方法"""
        db["key1"] = "value1"

        exists, value = db.key_exist("key1")
        assert exists
        assert value == "value1"

        exists, value = db.key_exist("nonexistent")
        assert not exists
        assert value is None
    
    def test_options_handling(self, db):
        """测试选项参数处理"""
        # 读取选项
        read_opts = ReadOptions()
        read_opts.fill_cache(True)
        value = db.get("key", options=read_opts)
        assert value is None
        
        # 写入选项
        write_opts = WriteOptions()
        write_opts.disable_wal = True  # 使用属性赋值
        db.put("key", "value", options=write_opts)
        assert db.get("key") == "value"

class TestIterationOperations:
    @pytest.fixture(autouse=True)
    def setup_logging(self, caplog):
        caplog.set_level(logging.INFO)

    @pytest.fixture
    def db_path(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)
    
    @pytest.fixture
    def db(self, db_path):
        db = BaseRocksDB(db_path)
        try:
            yield db
        finally:
            db.close()
    
    @pytest.fixture
    def populated_db(self, db):
        """创建并填充测试数据的数据库"""
        # 添加测试数据
        test_data = {
            "user:01": "alice",
            "user:02": "bob",
            "user:03": "david",
            "user:04": "emma",
            "user:05": "frank",
            "user:06": "grace",
            "user:07": "henry",
            "user:08": "iris",
            "user:09": "jack",
            "user:10": "kelly",
            "user:11": "lucas",
            "config:01": "setting1",
            "config:02": "setting2",
            "log:01": "log entry 1",
            "log:02": "log entry 2",
        }
        for k, v in test_data.items():
            db.put(k, v)
            logger.info(f"Added test data: {k} = {v}")
        
        # 验证数据写入
        for k, v in test_data.items():
            stored_v = db.get(k)
            logger.info(f"Verified data: {k} = {stored_v}")
            assert stored_v == v
            
        return db
    
    def test_iter_basic(self, populated_db):
        """测试基本迭代功能"""
        # 全量迭代
        items = list(populated_db.iter())
        assert len(items) == 15
        
        # 前缀迭代
        user_items = list(populated_db.iter(prefix="user:"))
        assert len(user_items) == 11
        assert all(k.startswith("user:") for k, _ in user_items)
    
    def test_iter_range(self, populated_db):
        """测试范围迭代"""
        # 清理已有数据
        for key in list(populated_db.iter()):
            del populated_db[key[0]]
        
        # 添加格式统一的测试数据
        test_data = {
            f"user:{i:02d}": f"user_{i:02d}" 
            for i in range(1, 11)  # 生成 user:01 到 user:10
        }
        
        for k, v in test_data.items():
            populated_db[k] = v
            logger.info(f"Added test data: {k} = {v}")
        
        # 验证所有数据
        all_items = list(populated_db.iter())
        logger.info(f"All items in db: {all_items}")
        
        # 测试用例：(start, end, expected_count, description)
        test_cases = [
            ("user:03", "user:05", 2, "区间 [user:03, user:05)"),
            (None, "user:03", 2, "上界 [user:01, user:03)"),
            ("user:08", None, 3, "下界 [user:08, ...)"),
            ("user:99", None, 0, "无匹配的下界"),
            (None, "user:00", 0, "无匹配的上界"),
            ("user:05", "user:05", 0, "空区间"),
        ]
        
        for start, end, expected_count, desc in test_cases:
            logger.info(f"\nTesting: {desc}")
            items = list(populated_db.iter(start=start, end=end))
            logger.info(f"Found items: {items}")
            assert len(items) == expected_count, \
                f"Expected {expected_count} items for {desc}, got {len(items)}: {items}"
    
    def test_iter_reverse(self, populated_db):
        """测试反向迭代"""
        # 普通反向迭代
        items = list(populated_db.iter(reverse=True))
        assert len(items) == 15
        assert items[0][0] > items[-1][0]  # 确保降序
        
        # 带前缀的反向迭代
        items = list(populated_db.iter(prefix="user:", reverse=True))
        assert len(items) == 11
        assert all(k.startswith("user:") for k, _ in items)
        assert items[0][0] > items[-1][0]  # 确保降序
        
        # 带范围的反向迭代
        items = list(populated_db.iter(start="user:02", end="user:05", reverse=True))
        assert len(items) == 3  # user:04, user:03, user:02
        assert items[0][0] == "user:04"
        assert items[-1][0] == "user:02"
    
    def test_iter_performance_options(self, populated_db):
        """测试性能相关选项"""
        # 测试 fill_cache=False 选项
        items = list(populated_db.iter(
            fill_cache=False,
        ))
        assert len(items) == 15  # 更新为实际的数据量
        
        # 测试自定义 ReadOptions
        opts = ReadOptions()
        opts.fill_cache(False)
        items = list(populated_db.iter(options=opts))
        assert len(items) == 15  # 更新为实际的数据量
        
        # 验证数据内容没有变化
        assert all(isinstance(k, str) for k, _ in items)
        assert all(isinstance(v, str) for _, v in items)
    
    def test_keys_values_items(self, populated_db):
        """测试键值获取方法"""
        # 测试 keys
        keys = populated_db.keys(prefix="user:")
        assert len(keys) == 11
        assert all(k.startswith("user:") for k in keys)
        assert "user:01" in keys
        assert "user:11" in keys
        
        # 测试 values
        values = populated_db.values(prefix="user:")
        assert len(values) == 11
        assert "alice" in values
        assert "lucas" in values
        
        # 测试 items
        items = populated_db.items(prefix="user:")
        assert len(items) == 11
        assert ("user:01", "alice") in items
        assert ("user:11", "lucas") in items

        # 测试 items
        items = populated_db.items(prefix="user:", limit=1)
        logger.info(f"Items with limit 1: {items}")
        assert len(items) == 1
        assert ("user:01", "alice") in items

    def test_iter_keys_values(self, populated_db):
        """测试键值迭代器"""
        # 测试键迭代器
        keys = list(populated_db.iter_keys(prefix="log:"))
        assert len(keys) == 2
        assert all(k.startswith("log:") for k in keys)
        
        # 测试值迭代器
        values = list(populated_db.iter_values(prefix="log:"))
        assert len(values) == 2
        assert all(v.startswith("log entry") for v in values)
    
    def test_edge_cases(self, populated_db):
        """测试边界情况"""
        # 空前缀
        items = list(populated_db.iter(prefix=""))
        logger.info(f"Empty prefix returned: {items}")
        assert len(items) == 15
        
        # 不存在的前缀
        items = list(populated_db.iter(prefix="nonexistent:"))
        logger.info(f"Nonexistent prefix returned: {items}")
        assert len(items) == 0
        
        # 范围边界
        items = list(populated_db.iter(start="a", end="z"))
        logger.info(f"Range [a, z] returned: {items}")
        assert len(items) == 15

        # 范围边界
        items = list(populated_db.iter(start="a"))
        logger.info(f"Range [a, z] returned: {items}")
        assert len(items) == 15

        # 反向范围
        items = list(populated_db.iter(start="z", end="a", reverse=True))
        logger.info(f"Reverse range [z, a] returned: {items}")
        assert len(items) == 15

class TestColumnFamilies:
    @pytest.fixture(autouse=True)
    def setup_logging(self, caplog):
        caplog.set_level(logging.INFO)

    @pytest.fixture
    def db_path(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)
    
    @pytest.fixture
    def db(self, db_path):
        db = BaseRocksDB(db_path)
        try:
            yield db
        finally:
            db.close()
    
    def test_column_family_basic(self, db):
        """测试列族基本操作"""
        # 创建列族
        users_cf = db.create_column_family("users")
        assert "users" in db.list_column_families(db.path)
        
        # 获取列族
        users_cf2 = db.get_column_family("users")
        assert users_cf2 is not None
        
        # 写入和读取数据
        users_cf.put("user:01", "alice")
        assert users_cf.get("user:01") == "alice"
        
        # 删除列族
        db.drop_column_family("users")
        assert "users" not in db.list_column_families(db.path)
    
    def test_default_column_family(self, db):
        """测试默认列族"""
        default_cf = db.default_cf
        assert default_cf is not None
        
        # 写入和读取数据
        default_cf.put("key1", "value1")
        assert default_cf.get("key1") == "value1"
        assert db.get("key1") == "value1"  # 通过主DB实例也能读取

class TestBatchOperations:
    @pytest.fixture(autouse=True)
    def setup_logging(self, caplog):
        caplog.set_level(logging.INFO)

    @pytest.fixture
    def db_path(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)
    
    @pytest.fixture
    def db(self, db_path):
        db = BaseRocksDB(db_path)
        try:
            yield db
        finally:
            db.close()
    
    def test_batch_basic(self, db):
        """测试基本批处理操作"""
        # 创建批处理
        batch = WriteBatch()
        
        # 添加操作
        batch.put("key1", "value1")
        batch.put("key2", "value2")
        batch.delete("key1")  # 删除操作
        
        # 执行批处理
        db.write(batch)
        
        # 验证结果
        assert db.get("key2") == "value2"

        exists, value = db.key_exist("key1")
        assert not exists
        assert value is None
    
    def test_batch_with_column_family(self, db):
        """测试在批处理中使用列族"""
        # 创建列族
        users_cf = db.create_column_family("users")
        posts_cf = db.create_column_family("posts")
        
        # 获取列族句柄
        users_handle = db.get_column_family_handle("users")
        posts_handle = db.get_column_family_handle("posts")
        
        # 创建批处理
        batch = WriteBatch()
        
        # 添加不同列族的操作
        batch.put("user:01", "alice", users_handle)
        batch.put("user:02", "bob", users_handle)
        batch.put("post:01", "Hello World", posts_handle)
        
        # 执行批处理
        db.write(batch)
        
        # 验证结果
        assert users_cf.get("user:01") == "alice"
        assert users_cf.get("user:02") == "bob"
        assert posts_cf.get("post:01") == "Hello World"
        
        # 清理
        db.drop_column_family("users")
        db.drop_column_family("posts")
    
    def test_batch_rollback(self, db):
        """测试批处理回滚"""
        # 预先写入一些数据
        db.put("key1", "original1")
        db.put("key2", "original2")
        
        try:
            batch = WriteBatch()
            batch.put("key1", "new1")
            batch.put("key2", "new2")
            raise Exception("Simulated error")  # 模拟错误
            db.write(batch)  # 这行不会执行
        except Exception:
            pass  # 预期的异常
        
        # 验证数据没有改变
        assert db.get("key1") == "original1"
        assert db.get("key2") == "original2"

    def test_read_only_mode(self, db_path):
        """测试只读模式（同一进程）
        
        1. 写入数据
        2. 关闭数据库
        3. 以只读模式重新打开
        4. 验证可以读取数据
        """
        # 写入数据
        db = BaseRocksDB(db_path)
        test_data = {
            "key1": "value1",
            "key2": "value2",
            "user:01": "alice"
        }
        for k, v in test_data.items():
            db.put(k, v)
        db.close()
        
        # 以只读模式打开
        from rocksdict import AccessType
        ro_db = BaseRocksDB(db_path, access_type=AccessType.read_only())
        
        # 验证数据可读
        for k, expected in test_data.items():
            assert ro_db.get(k) == expected
        
        # 验证写入失败
        try:
            ro_db.put("new_key", "new_value")
            assert False, "只读模式不应允许写入"
        except Exception as e:
            logger.info(f"预期中的写入错误: {e}")
        finally:
            ro_db.close()

    def test_secondary_mode(self, db_path):
        """测试主从模式（同一进程）
        
        1. 写入初始数据并保持数据库打开
        2. 以从库模式打开另一个实例
        3. 在主库写入新数据
        4. 验证从库可以通过同步读取到新数据
        """
        import os
        
        # 写入初始数据
        main_db = BaseRocksDB(db_path)
        main_db.put("key1", "value1")
        main_db.put("key2", "value2")
        
        # 准备从库路径
        secondary_path = os.path.join(db_path, "_secondary")
        os.makedirs(secondary_path, exist_ok=True)
        
        # 以从库模式打开另一个实例
        from rocksdict import AccessType, Rdict
        sec_db = Rdict(db_path, access_type=AccessType.secondary(secondary_path))
        
        # 验证可以读取初始数据
        assert sec_db.get("key1") == "value1"
        assert sec_db.get("key2") == "value2"
        
        # 主库写入新数据
        main_db.put("key3", "value3")
        
        # 从库同步并验证数据
        sec_db.try_catch_up_with_primary()
        
        # 验证从库可以读取所有数据（包括新增的）
        assert sec_db.get("key1") == "value1"
        assert sec_db.get("key2") == "value2" 
        assert sec_db.get("key3") == "value3"
        
        # 验证从库不能写入
        try:
            sec_db.put("key4", "value4")
            assert False, "从库模式不应允许写入"
        except Exception as e:
            logger.info(f"预期中的写入错误: {e}")
        finally:
            # 清理资源
            sec_db.close()
            main_db.close()

class TestPaginateMethod:
    @pytest.fixture(autouse=True)
    def setup_logging(self, caplog):
        caplog.set_level(logging.INFO)

    @pytest.fixture
    def db_path(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    def test_paginate_method(self, db_path):
        """测试新增的paginate方法
        
        验证paginate方法的基本功能：
        1. 获取第一页时不提供游标
        2. 使用返回的next_cursor获取后续页面
        3. 验证数据完整性和分页正确性
        """
        # 准备测试环境和数据
        db = BaseRocksDB(db_path)
        
        # 插入100条有序测试数据
        total_items = 100
        page_size = 10
        
        # 使用有序键，确保排序可预测
        for i in range(total_items):
            key = f"item:{i:03d}"
            value = f"value-{i:03d}"
            db.put(key, value)
        
        # 模拟分页查询
        all_pages_items = []
        current_page = 1
        next_cursor = None
        
        logger.info(f"开始使用paginate方法进行分页测试，页大小: {page_size}")
        
        while True:
            # 使用paginate方法获取当前页
            result = db.paginate(
                prefix='item:', 
                page_size=page_size,
                cursor=next_cursor
            )
            
            page_items = result['items']
            next_cursor = result['next_cursor']
            has_more = result['has_more']
            
            # 如果没有更多数据，退出循环
            if not page_items or not has_more:
                if not has_more:
                    logger.info(f"分页查询完成，总共 {current_page} 页")
                    # 添加最后一页的数据
                    all_pages_items.extend(page_items)
                    break
            
            logger.info(f"第 {current_page} 页: {[k for k, _ in page_items]}")
            
            # 保存结果用于验证
            all_pages_items.extend(page_items)
            
            # 下一页
            current_page += 1
            
            # 如果没有下一个游标，退出循环
            if not next_cursor:
                break
        
        # 验证所有数据都被正确分页
        assert len(all_pages_items) == total_items
        
        # 验证数据完整性和顺序
        for i, (key, value) in enumerate(all_pages_items):
            expected_key = f"item:{i:03d}"
            expected_value = f"value-{i:03d}"
            assert key == expected_key, f"位置 {i} 的键应为 {expected_key}，实际是 {key}"
            assert value == expected_value, f"位置 {i} 的值应为 {expected_value}，实际是 {value}"
        
        # 验证分页数量正确
        expected_pages = (total_items + page_size - 1) // page_size  # 向上取整
        assert current_page == expected_pages, f"应该有 {expected_pages} 页，实际有 {current_page}"
        
        # 测试反向分页
        reverse_result = db.paginate(
            prefix='item:', 
            page_size=5,
            reverse=True
        )
        
        # 验证反向分页获取的是最后的数据
        reverse_items = reverse_result['items']
        assert len(reverse_items) == 5
        
        # 最后5个项目应该是反序的
        for i, (key, value) in enumerate(reverse_items):
            expected_index = total_items - 1 - i
            expected_key = f"item:{expected_index:03d}"
            assert key == expected_key, f"反向查询第 {i} 项应为 {expected_key}，实际是 {key}"
        
        db.close()
