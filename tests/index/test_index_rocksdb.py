import pytest
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
from voidring.index import IndexedRocksDB
import tempfile
import shutil
import time
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def setup_logging(caplog):
    caplog.set_level(logging.INFO)

@pytest.fixture
def db_path():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

@pytest.fixture
def db(db_path):
    db = IndexedRocksDB(db_path)
    try:
        yield db
    finally:
        db.close()

# 测试用的数据模型
class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

    @property
    def mykey(self):
        return f"{self.name}_{self.email}"
    
class Post(BaseModel):
    title: str
    author: str
    tags: List[str]
    created_at: datetime
    metadata: Dict[str, str]

class TestBasicIndexOperations:
    """基础索引操作测试"""
        
    def test_register_collection_index(self, db):
        """测试注册集合字段索引"""
        db.register_collection("users", User)
        db.register_index("users", User, "name")
        db.register_index("users", User, "age")
        
        # 验证索引元数据
        metadata = list(db.indexes_metadata_cf.items())
        assert len(metadata) == 3

        with pytest.raises(ValueError) as e:
            db.register_index("users", User, "sex")
        assert "无效的" in str(e.value)

    def test_basic_crud_with_index(self, db):
        """测试带索引的基本CRUD操作"""
        db.register_collection("users", User)
        db.register_index("users", User, "name")
        db.register_index("users", User, "mykey")
        
        # 创建
        user = User(name="alice", age=25)
        db.update_with_indexes("users", "user:1", user)

        # 查看所有Keys
        keys = list(db.iter_collection_keys("users"))
        logger.info(f"keys: {keys}")
        assert len(keys) == 1

        # 读取
        user_data = db.get("user:1")
        assert user_data["name"] == "alice"
        assert user_data["age"] == 25
        
        # 或使用get_as_model
        user_model = db.get_as_model("users", "user:1")
        assert user_model == user
        
        # 通过索引查询
        keys = list(db.iter_keys_with_index("users", "name", "alice"))
        assert len(keys) == 1
        assert keys[0] == "user:1"
        
        # 更新
        user.name = "alice2"
        db.update_with_indexes("users", "user:1", user)
        keys = list(db.iter_keys_with_index("users", "name", "alice2"))
        assert len(keys) == 1
        
        # 删除
        db.delete_with_indexes("users", "user:1")
        keys = list(db.iter_keys_with_index("users", "name", "alice2"))
        assert len(keys) == 0

    def test_rebuild_indexes(self, db):
        """测试重建索引"""
        # 只有注册集合才能重建索引
        db.register_collection("users", User)

        db.update_with_indexes("users", "user:1", User(name="alice", age=25))
        db.update_with_indexes("users", "user:2", User(name="bob", age=30))
        db.update_with_indexes("users", "user:3", User(name="alice", age=26))

        # 之前没有注册索引，因此更新时不会自动建立，无法查询
        items = db.items_with_index(
            collection_name="users",
            field_path="name",
            field_value="bob"
        )
        assert len(items) == 0

        # 注册索引，然后重建
        db.register_index("users", User, "name")
        db.rebuild_indexes("users")
        # 现在就可以查询了
        items = db.items_with_index(
            collection_name="users",
            field_path="name",
            field_value="bob"
        )
        logger.info(f"items: {items}")
        assert len(items) == 1
        assert items[0][0] == "user:2"
        # 转换回模型对象进行比较
        user = User.model_validate(items[0][1])
        assert user == User(name="bob", age=30)

    def test_simple_type_index(self, db):
        """典型的字典索引"""
        db.register_index("names", dict, "name")
        db.update_with_indexes(collection_name="names", key="alice", value={"name": "Alice Xue"})
        items = db.items_with_index(collection_name="names", field_path="name", field_value="Alice Xue")
        logger.info(f"items: {items}")
        assert len(items) == 1
        assert items[0][0] == "alice"
        assert items[0][1] == {"name": "Alice Xue"}

    def test_null_index(self, db):
        """空索引"""
        db.register_index("names", str, "")
        db.update_with_indexes(collection_name="names", key="alice", value="Alice Xue")
        items = db.items_with_index(collection_name="names", field_path="", field_value="Alice Xue")
        logger.info(f"items: {items}")
        assert len(items) == 1
        assert items[0][0] == "alice"
        assert items[0][1] == "Alice Xue"

    def test_dict_value_index(self, db):
        """索引字典值"""
        db.register_index("names", dict, "name")
        value = {"last": "Xue", "first": "Alice"}
        db.update_with_indexes(collection_name="names", key="alice", value={"name": value})
        items = db.items_with_index(collection_name="names", field_path="name", field_value=value)
        logger.info(f"items: {items}")
        assert len(items) == 1
        assert items[0][0] == "alice"
        assert items[0][1] == {"name": value}

    def test_property_index(self, db):
        """索引属性"""
        db.register_collection("users", User)
        db.register_index("users", User, "email")
        db.register_index("users", User, "mykey")
        
        # 创建
        user = User(name="alice", age=25, email="alice@example.com")
        db.update_with_indexes("users", "user:1", user)

        # 查看索引
        cf = db.get_column_family(db.INDEX_CF)
        keys = list(db.iter(rdict=cf, prefix="idx"))
        logger.info(f"indexes: {keys}")

        # 查看所有Keys
        keys = list(db.iter_collection_keys("users"))
        logger.info(f"keys: {keys}")
        assert len(keys) == 1

        # 读取并转换回模型
        stored_user = db.get_as_model("users", "user:1")
        assert stored_user == user
        
        # 通过索引查询
        keys = list(db.iter_keys_with_index("users", "mykey", "alice_alice@example.com"))
        assert len(keys) == 1
        assert keys[0] == "user:1"
        

class TestComplexIndexOperations:
    """复杂索引操作测试"""
    
    def test_dict_field_index(self, db):
        post = {
            "title": "test",
            "author": "alice",
            "tags": ["python", "test"],
            "created_at": datetime.now(),
            "metadata": {"category": "tech"}
        }
        PostType = Dict[str, Union[
            str,                    # 用于 title 和 author
            List[str],             # 用于 tags
            datetime,              # 用于 created_at
            Dict[str, str]         # 用于 metadata
        ]]

        db.register_index("posts", PostType, "metadata.category")
        db.update_with_indexes("posts", "post:1", post)
        items = db.keys(prefix="idx", rdict=db.get_column_family(db.INDEX_CF))
        logger.info(f"index items: {items}")
        
        keys = list(db.iter_keys_with_index("posts", "metadata.category", "tech"))
        logger.info(f"keys: {keys}")
        assert len(keys) == 1
        assert keys[0] == "post:1"

    def test_model_field_index(self, db):
        """测试嵌套字段索引"""
        post = Post(
            title="test",
            author="alice",
            tags=["python", "test"],
            created_at=datetime.now(),
            metadata={"category": "tech"}
        )
        
        db.register_index("posts", Post, "metadata.category")
        db.update_with_indexes("posts", "post:1", post)
        
        keys = list(db.iter_keys_with_index("posts", "metadata.category", "tech"))
        assert len(keys) == 1
        assert keys[0] == "post:1"

class TestRangeQueries:
    """范围查询测试"""
    
    @pytest.fixture
    def db_with_data(self, db_path):
        db = IndexedRocksDB(db_path)
        db.register_index("users", User, "age")
        
        # 插入测试数据
        for i in range(10):
            user = User(name=f"user{i}", age=20+i)
            db.update_with_indexes("users", f"user:{i}", user)
            
        try:
            yield db
        finally:
            db.close()
    
    def test_range_query(self, db_with_data):
        """测试范围查询"""
        # 查询年龄在 22-25 之间的用户
        # 查询方法遵循左闭右开的原则
        keys = list(db_with_data.iter_keys_with_index(
            "users", "age", 
            start=22, 
            end=26
        ))
        assert len(keys) == 4
        
        # 反向查询
        keys_reverse = list(db_with_data.iter_keys_with_index(
            "users", "age", 
            start=26, 
            end=22, 
            reverse=True
        ))
        assert len(keys_reverse) == 4
        assert sorted(keys_reverse) == sorted(keys)

        items = list(db_with_data.items_with_index(
            collection_name="users",
            field_path="age", 
            start=22, 
            end=26
        ))
        assert len(items) == 4

        keys = list(db_with_data.keys_with_index(
            collection_name="users",
            field_path="age", 
            start=22, 
            end=26
        ))
        assert len(keys) == 4

        values = list(db_with_data.values_with_index(
            collection_name="users",
            field_path="age", 
            start=22, 
            end=26
        ))
        assert len(values) == 4

class TestSpecialCases:
    """特殊情况测试"""
    
    @pytest.fixture
    def db(self, db_path):
        db = IndexedRocksDB(db_path)
        try:
            yield db
        finally:
            db.close()
    
    def test_null_values(self, db):
        """测试空值索引"""
        db.register_index("users", User, "email")
        
        user1 = User(name="alice", age=25)  # email is None
        user2 = User(name="bob", age=30, email="bob@example.com")
        
        db.update_with_indexes("users", "user:1", user1)
        db.update_with_indexes("users", "user:2", user2)
        
        # 查询 email 为空的用户
        keys = list(db.iter_keys_with_index("users", "email", None))
        assert len(keys) == 1
        assert keys[0] == "user:1"
    
    def test_special_characters(self, db):
        """测试特殊字符处理"""
        db.register_index("users", User, "name")
        
        user = User(name="test:user.with/special*chars", age=25)
        db.update_with_indexes("users", "user:1", user)
        
        keys = list(db.iter_keys_with_index(
            "users", "name", "test:user.with/special*chars"
        ))
        assert len(keys) == 1
        assert keys[0] == "user:1"

class TestPydanticCompatibility:
    """Pydantic模型兼容性测试"""
    
    # 创建定义相同但名称不同的类
    class Customer(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
        
    # 扩展的模型，包含额外字段
    class ExtendedUser(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
        phone: Optional[str] = None  # 新增字段
        address: Optional[str] = None  # 新增字段
        
    # 简化的模型，字段较少
    class SimpleUser(BaseModel):
        name: str
        age: int
    
    def test_same_structure_different_class(self, db):
        """测试结构相同但名称不同的类加载数据"""
        # 使用User保存数据
        db.register_collection("people", User)
        db.register_index("people", User, "name")
        
        user = User(name="alice", age=25, email="alice@example.com")
        db.update_with_indexes("people", "person:1", user)
        
        # 使用结构相同但不同名称的类加载
        stored_data = db.get("person:1")
        customer = self.Customer.model_validate(stored_data)
        
        assert customer.name == "alice"
        assert customer.age == 25
        assert customer.email == "alice@example.com"
        
        # 使用get_as_model直接指定模型类
        customer = db.get_as_model("people", "person:1", model_class=self.Customer)
        assert isinstance(customer, self.Customer)
        assert customer.name == "alice"
    
    def test_compatible_model_fields(self, db):
        """测试字段兼容性的模型加载数据"""
        db.register_collection("users", User)
        
        # 保存标准User数据
        user = User(name="alice", age=25, email="alice@example.com")
        db.update_with_indexes("users", "user:1", user)
        
        # 使用扩展模型读取数据(多字段模型读取少字段数据)
        extended = db.get_as_model("users", "user:1", model_class=self.ExtendedUser)
        assert extended.name == "alice"
        assert extended.age == 25
        assert extended.email == "alice@example.com"
        assert extended.phone is None  # 额外字段为None
        
        # 使用简化模型读取数据(少字段模型读取多字段数据)
        simple = db.get_as_model("users", "user:1", model_class=self.SimpleUser)
        assert simple.name == "alice"
        assert simple.age == 25
        # simple没有email字段，但仍能正常加载
        
        # 保存扩展模型数据
        extended = self.ExtendedUser(
            name="bob", 
            age=30, 
            email="bob@example.com",
            phone="123-456-7890",
            address="123 Main St"
        )
        db.update_with_indexes("users", "user:2", extended)
        
        # 使用标准模型读取扩展数据
        standard = db.get_as_model("users", "user:2", model_class=User)
        assert standard.name == "bob"
        assert standard.age == 30
        assert standard.email == "bob@example.com"
        # 额外字段(phone,address)被忽略但不影响读取
        
    def test_auto_model_conversion(self, db):
        """测试自动模型转换"""
        db.register_collection("users", User)
        db.register_index("users", User, "age")
        
        # 保存用户数据
        for i in range(3):
            user = User(name=f"user{i}", age=20+i, email=f"user{i}@example.com")
            db.update_with_indexes("users", f"user:{i}", user)
        
        # 使用return_as_model自动转换成模型
        items = db.items_with_index("users", "age", start=20, end=22, return_as_model=True)
        users = [item[1] for item in items]
        
        assert len(users) == 2
        assert all(isinstance(u, User) for u in users)
        assert users[0].name in ["user0", "user1"]
        
        # 指定不同的模型类
        items = db.items_with_index(
            "users", "age", 
            start=20, end=22, 
            model_class=self.Customer, 
            return_as_model=True
        )
        customers = [item[1] for item in items]
        
        assert len(customers) == 2
        assert all(isinstance(c, self.Customer) for c in customers)

class TestCollectionManagement:
    """集合管理功能测试"""
    
    def test_collection_registry(self, db):
        """测试集合注册表功能"""
        # 注册集合
        db.register_collection("users", User)
        db.register_index("users", User, "name")
        db.register_index("users", User, "email")
        
        db.register_collection("posts", Post)
        db.register_index("posts", Post, "title")
        db.register_index("posts", Post, "metadata.category")
        
        # 获取所有集合
        collections = db.get_collections()
        assert len(collections) == 2
        
        # 集合名称集合应包含所有注册的集合
        collection_names = {c.name for c in collections}
        assert collection_names == {"users", "posts"}
        
        # 获取特定集合信息
        user_info = db.get_collection_info("users")
        assert user_info.name == "users"
        assert user_info.model_class == User
        assert len(user_info.field_paths) == 3  # name, email, #
        assert "name" in user_info.field_paths
        assert "email" in user_info.field_paths
        assert "#" in user_info.field_paths  # 自动添加的ID索引
        
        post_info = db.get_collection_info("posts")
        assert post_info.model_class == Post
        assert "title" in post_info.field_paths
        assert "metadata.category" in post_info.field_paths
    
    def test_collection_iteration(self, db):
        """测试集合迭代功能"""
        db.register_collection("users", User)
        
        # 插入一些用户
        for i in range(5):
            user = User(name=f"user{i}", age=20+i)
            db.update_with_indexes("users", f"user:{i}", user)
        
        # 测试迭代集合，明确指定不转换模型
        count = 0
        for key, value in db.iter_collection("users", return_as_model=False):
            assert key.startswith("user:")
            assert isinstance(value, dict)
            assert value["name"].startswith("user")
            count += 1
        
        assert count == 5
        
        # 测试自动转换为模型
        models = []
        for key, value in db.iter_collection("users", return_as_model=True):
            assert isinstance(value, User)
            models.append(value)
        
        assert len(models) == 5
        assert all(isinstance(m, User) for m in models)
        
    def test_collection_persistence(self, db_path):
        """测试集合信息在实例重启后的恢复"""
        # 为每个实例使用单独的子目录
        db1_path = os.path.join(db_path, "db1")
        os.makedirs(db1_path, exist_ok=True)
        
        # 第一个数据库实例
        db1 = IndexedRocksDB(db1_path)
        try:
            db1.register_collection("users", User)
            db1.register_index("users", User, "name")
            
            # 插入测试数据
            db1.update_with_indexes("users", "user:1", User(name="alice", age=25))
        finally:
            db1.close()
        
        # 确保第一个实例完全释放资源
        db1 = None
        
        # 验证数据持久化 - 创建完全相同的新路径
        # 在实际应用中这是同一路径，但在测试中我们复制数据
        import shutil
        db2_path = os.path.join(db_path, "db2") 
        shutil.copytree(db1_path, db2_path)
        
        # 创建新的数据库实例(模拟重启)
        db2 = IndexedRocksDB(db2_path)
        try:
            # 重新注册相同集合(通常在应用启动时)
            db2.register_collection("users", User)
            db2.register_index("users", User, "name")
            
            # 验证可以使用索引查询
            keys = list(db2.iter_keys_with_index("users", "name", "alice"))
            assert len(keys) == 1
            assert keys[0] == "user:1"
            
            # 使用get_as_model获取模型
            user = db2.get_as_model("users", "user:1")
            assert isinstance(user, User)
            assert user.name == "alice"
        finally:
            db2.close() 