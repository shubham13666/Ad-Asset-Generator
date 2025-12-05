from tencentcloud.hunyuan.v20230901 import models
print("✅ Available Request classes:")
for name in dir(models):
    if 'Request' in name:
        print(f"  - {name}")

print("\n✅ Available Client methods:")
from tencentcloud.hunyuan.v20230901 import hunyuan_client
print(dir(hunyuan_client.HunyuanClient))
