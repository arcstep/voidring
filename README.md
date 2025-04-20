# VoidRing

VoidRing是一个基于RocksDB的高性能键值存储引擎，具有空间压缩能力和丰富的功能扩展。它提供了友好的Python接口，使得在Python应用中使用RocksDB变得简单而强大。

## 特性

- **高性能存储**：基于RocksDB的LSM树结构，提供高吞吐量的读写操作
- **自动索引**：支持对Pydantic模型和普通字典的多字段索引
- **游标分页**：高效的基于游标的分页查询，适用于大数据集
- **多种访问模式**：支持读写、只读、带TTL和主从复制等多种访问模式
- **类型安全**：与Pydantic模型无缝集成，提供类型安全的数据存储

## 安装

```bash
pip install voidring
```

## RocksDB优势与VoidRing增强

### RocksDB优势
- **高写入性能**：基于LSM树的设计，写入操作极快
- **可调优性**：提供丰富的配置选项，适应不同场景
- **空间效率**：支持高效的数据压缩
- **稳定可靠**：经Facebook、LinkedIn等公司生产环境验证

### VoidRing增强
- **Python友好接口**：提供符合Python习惯的API
- **自动索引系统**：支持复杂对象和嵌套字段的索引
- **Pydantic集成**：自动处理Pydantic模型的序列化与反序列化
- **高级查询功能**：提供游标分页、范围查询等功能
- **多种访问模式封装**：简化只读、主从复制等高级功能的使用

## 典型用例

```python
from voidring.index import IndexedRocksDB
from pydantic import BaseModel
from typing import Optional

# 定义数据模型
class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

# 创建带索引的数据库
db = IndexedRocksDB("./data/userdb")

# 注册集合和索引
db.register_collection("users", User)
db.register_index("users", User, "name")
db.register_index("users", User, "age")

# 写入数据（自动更新索引）
user = User(name="张三", age=30, email="zhangsan@example.com")
db.update_with_indexes("users", "user:1", user)

# 通过索引查询
for key, user in db.items_with_index(
    collection_name="users",
    field_path="name",
    field_value="张三"
):
    print(f"找到用户: {user}")

# 范围查询（查找25-35岁的用户）
for key, user in db.items_with_index(
    collection_name="users",
    field_path="age",
    start=25,
    end=36
):
    print(f"年龄在范围内: {user}")
```

## 分页查询

```python
from voidring.index import IndexedRocksDB
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    
db = IndexedRocksDB("./data/userdb")
db.register_collection("users", User)
db.register_index("users", User, "age")

# 插入测试数据
for i in range(100):
    db.update_with_indexes(
        "users", 
        f"user:{i}", 
        User(name=f"用户{i}", age=20 + i % 5)
    )

# 查询所有年龄为22的用户（分页）
result = db.paginate_with_index(
    collection_name="users",
    field_path="age",
    field_value=22,
    page_size=5,
    return_as_model=True
)

# 显示结果
for key, user in result["items"]:
    print(f"{key}: {user.name}, 年龄: {user.age}")

# 获取下一页
if result["has_more"]:
    next_cursor = result["next_cursor"]
    print(f"获取下一页: {next_cursor}")
```

## 性能建议

- 对于大批量写入，使用批处理功能而不是单独的put
- 为频繁查询的字段创建索引
- 使用适当的压缩选项平衡性能和空间使用
- 对于只读场景，使用只读模式提高并发性能
- 使用游标分页处理大型结果集，避免一次加载过多数据

## 贡献

欢迎提交问题报告、功能请求和PR至项目仓库。
