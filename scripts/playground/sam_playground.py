import os
import dotenv
from transformers import SamModel, SamProcessor, set_seed

set_seed(42)
cache_dir = ".model.cache"
# NOTE(xiaoke): load sas_key from .env for huggingface model downloading.
dotenv.load_dotenv(".env")
use_auth_token = os.getenv("USE_AUTH_TOKEN", False)

# sam_model_name = "facebook/sam-vit-base"
# sam_model_name = "facebook/sam-vit-large"
sam_model_name = "facebook/sam-vit-huge"
model = SamModel.from_pretrained(
    sam_model_name,
    cache_dir=cache_dir,
    use_auth_token=use_auth_token,
)
processor = SamProcessor.from_pretrained(
    sam_model_name,
    device_map="auto",
    torch_dtype="auto",
    cache_dir=cache_dir,
    use_auth_token=use_auth_token,
)
