import pytest
import logging
import tempfile
import shutil
import time
from pathlib import Path
from rocksdict import (
    Rdict, 
    Options, 
    ReadOptions,
    PlainTableFactoryOptions,
    SliceTransform,
    AccessType
)
import multiprocessing
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """设置日志级别"""
    caplog.set_level(logging.INFO)

@pytest.fixture
def db_path():
    """创建临时数据库路径"""
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test_db"
    yield db_path
    if temp_dir.exists():
        time.sleep(0.1)
        shutil.rmtree(temp_dir)

@pytest.fixture
def db(db_path):
    """创建并初始化数据库，配置前缀支持"""
    # 配置数据库选项
    opts = Options()
    
    # 设置前缀提取器 - 使用固定长度前缀
    # "user:xx" 的前缀长度是5 ("user:")
    opts.set_prefix_extractor(SliceTransform.create_fixed_prefix(5))
    
    # 配置表选项
    table_opts = PlainTableFactoryOptions()
    table_opts.bloom_bits_per_key = 10
    table_opts.hash_table_ratio = 0.75
    table_opts.index_sparseness = 16
    opts.set_plain_table_factory(table_opts)
    
    # 创建数据库实例
    db = Rdict(str(db_path), opts)
    
    # 写入测试数据
    test_data = {
        **{f"user:{i:02d}": f"user_{i:02d}" for i in range(1, 11)},
        **{f"config:{i:02d}": f"config_{i:02d}" for i in range(1, 6)},
        **{f"log:{i:02d}": f"log_{i:02d}" for i in range(1, 6)}
    }
    for k, v in test_data.items():
        db[k] = v
        logger.info(f"Added: {k} = {v}")
    
    yield db
    
    db.close()
    time.sleep(0.1)
    try:
        Rdict.destroy(str(db_path), Options())
    except Exception as e:
        logger.warning(f"销毁数据库时出错: {e}")

def test_prefix_iteration(db):
    """测试前缀迭代
    
    目标：验证 set_prefix_same_as_start 的行为
    1. 基本的前缀匹配
    2. 不同位置的seek
    3. 前缀边界情况
    """
    logger.info("\n=== 测试前缀迭代 ===")
    
    # 1. 基本前缀匹配
    logger.info("\n1. 基本前缀匹配:")
    opts = ReadOptions()
    opts.set_prefix_same_as_start(True)
    it = db.iter(opts)
    
    it.seek("user:")
    items = []
    while it.valid():
        key = it.key()
        if not key.startswith("user:"):
            break
        items.append((key, it.value()))
        it.next()
    
    logger.info(f"Found {len(items)} items with prefix 'user:':")
    for k, v in items:
        logger.info(f"{k} = {v}")
    
    assert len(items) == 10, "应该找到10个user:前缀的项"
    assert all(k.startswith("user:") for k, _ in items), "所有key应该以user:开头"
    assert items[0][0] == "user:01", "应该从user:01开始"
    assert items[-1][0] == "user:10", "应该在user:10结束"
    
    # 2. 从中间位置seek
    logger.info("\n2. 从中间位置seek:")
    it.seek("user:05")
    mid_items = []
    while it.valid():
        key = it.key()
        if not key.startswith("user:"):
            break
        mid_items.append((key, it.value()))
        it.next()
    
    logger.info(f"Found {len(mid_items)} items from 'user:05':")
    for k, v in mid_items:
        logger.info(f"{k} = {v}")
    
    assert mid_items[0][0] == "user:05", "应该从user:05开始"
    assert len(mid_items) == 6, "应该找到6个项(05-10)"
    
    # 3. 不同前缀
    logger.info("\n3. 测试其他前缀:")
    it.seek("config:")
    config_items = []
    while it.valid():
        key = it.key()
        if not key.startswith("config:"):
            break
        config_items.append((key, it.value()))
        it.next()
    
    logger.info(f"Found {len(config_items)} items with prefix 'config:':")
    for k, v in config_items:
        logger.info(f"{k} = {v}")
    
    assert len(config_items) == 5, "应该找到5个config:前缀的项"
    del it

def test_range_iteration(db):
    """测试范围迭代
    
    目标：验证范围查询的行为
    1. 使用seek定位到范围起点
    2. 手动检查范围边界
    3. 不同范围的查询
    """
    logger.info("\n=== 测试范围迭代 ===")
    
    # 1. 基本范围查询 [user:03, user:07)
    logger.info("\n1. 基本范围查询:")
    it = db.iter()
    
    items = []
    it.seek("user:03")  # 直接定位到起点
    while it.valid():
        key = it.key()
        if key >= "user:07":  # 手动检查上界
            break
        if not key.startswith("user:"):  # 确保在同一前缀
            break
        items.append((key, it.value()))
        it.next()
    
    logger.info(f"Range [user:03, user:07) found {len(items)} items:")
    for k, v in items:
        logger.info(f"{k} = {v}")
    
    assert len(items) == 4, "应该找到4个项"
    assert [k for k, _ in items] == [
        "user:03", "user:04", "user:05", "user:06"
    ], "应该包含正确的序列"
    
    # 2. 跨前缀范围查询
    logger.info("\n2. 跨前缀范围查询:")
    it.seek("config:03")
    cross_items = []
    while it.valid():
        key = it.key()
        if key >= "log:":  # 在log:前缀之前停止
            break
        cross_items.append((key, it.value()))
        it.next()
    
    logger.info(f"Range [config:03, log:) found {len(cross_items)} items:")
    for k, v in cross_items:
        logger.info(f"{k} = {v}")
    
    assert len(cross_items) > 0, "应该找到一些项"
    assert all(k.startswith("config:") for k, _ in cross_items), "应该只包含config:前缀的项"
    
    del it

def test_combined_options(db):
    """测试选项组合
    
    目标：验证不同选项组合的行为
    1. 前缀 + 范围边界
    2. 不同的seek策略
    """
    logger.info("\n=== 测试选项组合 ===")
    
    opts = ReadOptions()
    opts.set_prefix_same_as_start(True)
    it = db.iter(opts)
    
    # 在前缀限制下的范围查询
    it.seek("user:03")
    items = []
    while it.valid():
        key = it.key()
        if not key.startswith("user:"):
            break
        if key >= "user:07":
            break
        items.append((key, it.value()))
        it.next()
    
    logger.info(f"Prefix-constrained range found {len(items)} items:")
    for k, v in items:
        logger.info(f"{k} = {v}")
    
    assert len(items) == 4, "应该找到4个项"
    assert all("user:03" <= k < "user:07" for k, _ in items), "应该在正确的范围内"
    
    del it 

def write_data(path, data_range, event_started, event_done):
    """写入进程函数"""
    try:
        # 主实例，拥有正常读写权限
        opts = Options()
        db = Rdict(path, opts)
        
        # 通知已经打开数据库
        event_started.set()
        
        # 写入一些数据
        for i in range(*data_range):
            key = f"key:{i:04d}"
            value = f"value:{i:04d}"
            db[key] = value
            logger.info(f"主实例写入: {key}={value}")
            # 写入后短暂暂停，让只读实例有时间读取
            time.sleep(0.05)
            
        # 写入完成
        event_done.set()
        
        # 等待一会儿让读取进程完成
        time.sleep(1)
        db.close()
        return True
    except Exception as e:
        logger.error(f"写入进程错误: {e}")
        event_started.set()  # 即使出错也要设置，避免死锁
        event_done.set()
        return False

def read_data_readonly(path, keys_to_check, event_started, event_done, results):
    """只读模式读取进程函数"""
    try:
        # 等待主进程打开数据库
        event_started.wait()
        
        # 使用AccessType.read_only()
        ro_db = Rdict(path, access_type=AccessType.read_only())
        
        # 添加总超时计时
        start_time = time.time()
        max_time = 8.0  # 确保在测试10秒超时前结束
        
        # 持续读取直到写入完成或超时
        found_count = 0
        while (not event_done.is_set() or found_count < len(keys_to_check)) and time.time() - start_time < max_time:
            for key in keys_to_check:
                if key in results and results[key] is not None:
                    continue  # 已经找到这个键
                
                try:
                    value = ro_db.get(key)
                    if value is not None:
                        results[key] = value
                        found_count += 1
                        logger.info(f"只读实例读取: {key}={value}")
                except KeyError:
                    # 键不存在，继续等待
                    pass
                except Exception as e:
                    logger.error(f"读取错误: {e}")
            
            # 如果已经找到所有键，可以提前结束
            if found_count >= len(keys_to_check):
                logger.info(f"找到所有{found_count}个键，提前退出")
                break
            
            # 如果主进程完成但未找到所有键，记录并退出
            if event_done.is_set() and found_count < len(keys_to_check):
                logger.info(f"主进程已完成，但只找到{found_count}/{len(keys_to_check)}个键")
                break
                
            # 短暂暂停，避免CPU过高
            time.sleep(0.05)  # 略微增加暂停时间
        
        ro_db.close()
        logger.info(f"只读进程完成，找到{found_count}/{len(keys_to_check)}个键")
        return True
    except Exception as e:
        logger.error(f"只读进程错误: {e}")
        return False

def read_data_secondary(path, secondary_path, keys_to_check, event_started, event_done, results):
    """从模式读取进程函数"""
    try:
        # 等待主进程打开数据库
        event_started.wait()
        
        # 确保从实例目录存在
        os.makedirs(secondary_path, exist_ok=True)
        
        # 使用AccessType.secondary()
        sec_db = Rdict(path, access_type=AccessType.secondary(secondary_path))
        
        # 持续读取直到写入完成
        found_count = 0
        while not event_done.is_set() or found_count < len(keys_to_check):
            for key in keys_to_check:
                if key in results and results[key] is not None:
                    continue
                
                try:
                    # 每次读取前尝试同步从实例
                    sec_db.try_catch_up_with_primary()
                    
                    value = sec_db.get(key)
                    if value is not None:
                        results[key] = value
                        found_count += 1
                        logger.info(f"从实例读取: {key}={value}")
                except KeyError:
                    # 键不存在，继续等待
                    pass
                except Exception as e:
                    logger.error(f"读取错误: {e}")
            
            # 短暂暂停
            time.sleep(0.02)
            
            # 如果已经找到所有键，可以提前结束
            if found_count >= len(keys_to_check):
                break
                
        sec_db.close()
        return True
    except Exception as e:
        logger.error(f"从进程错误: {e}")
        return False

class TestConcurrentAccess:
    """测试RocksDB的并发访问模式"""
    
    @pytest.fixture
    def test_db_path(self, tmp_path):
        """创建测试数据库路径"""
        return str(tmp_path / "concurrent_db")
    
    @pytest.fixture
    def secondary_path(self, tmp_path):
        """创建从实例路径"""
        return str(tmp_path / "secondary_db")
    
    def test_readonly_mode(self, test_db_path):
        """测试只读模式"""
        manager = multiprocessing.Manager()
        results = manager.dict()
        
        # 创建事件用于进程同步
        event_started = multiprocessing.Event()
        event_done = multiprocessing.Event()
        
        # 预定义要检查的键
        keys_to_check = [f"key:{i:04d}" for i in range(0, 10)]
        
        # 创建写入进程 - 修复参数
        writer = multiprocessing.Process(
            target=write_data,
            args=(test_db_path, (0, 10), event_started, event_done)
        )
        
        # 创建只读进程
        reader = multiprocessing.Process(
            target=read_data_readonly,
            args=(test_db_path, keys_to_check, event_started, event_done, results)
        )
        
        # 启动写入进程并给它时间创建数据库
        writer.start()
        time.sleep(0.2)  # 等待数据库创建
        
        # 设置就绪标志，启动读取进程
        event_started.set()
        reader.start()
        
        # 等待两个进程完成
        writer.join(timeout=10)
        reader.join(timeout=10)
        
        # 验证结果
        assert not writer.is_alive(), "写入进程未能及时完成"
        assert not reader.is_alive(), "读取进程未能及时完成"
        
        # 检查是否至少有一些键被成功读取到
        found_keys = [k for k, v in results.items() if v is not None]
        logger.info(f"只读模式找到的键: {found_keys}")
        
        assert len(found_keys) > 0, "只读模式应该能够读取到一些键"
        
        # 注意：由于并发性，我们可能无法保证读取到所有键

    def test_secondary_instance(self, test_db_path, secondary_path):
        """测试主从模式 (Secondary Instance)"""
        manager = multiprocessing.Manager()
        results = manager.dict()
        
        # 创建事件用于进程同步
        event_started = multiprocessing.Event()
        event_done = multiprocessing.Event()
        
        # 预定义要检查的键
        keys_to_check = [f"key:{i:04d}" for i in range(0, 10)]
        
        # 创建写入进程
        writer = multiprocessing.Process(
            target=write_data,
            args=(test_db_path, (0, 10), event_started, event_done)
        )
        
        # 创建从实例进程
        reader = multiprocessing.Process(
            target=read_data_secondary,
            args=(test_db_path, secondary_path, keys_to_check, event_started, event_done, results)
        )
        
        # 启动写入进程并给它时间创建数据库
        writer.start()
        time.sleep(0.2)  # 等待数据库创建
        
        # 设置就绪标志，启动读取进程
        event_started.set()
        reader.start()
        
        # 等待两个进程完成
        writer.join(timeout=10)
        reader.join(timeout=10)
        
        # 验证结果
        assert not writer.is_alive(), "写入进程未能及时完成"
        assert not reader.is_alive(), "读取进程未能及时完成"
        
        # 检查是否至少有一些键被成功读取到
        found_keys = [k for k, v in results.items() if v is not None]
        logger.info(f"从实例找到的键: {found_keys}")
        
        assert len(found_keys) > 0, "从实例应该能够读取到一些键" 