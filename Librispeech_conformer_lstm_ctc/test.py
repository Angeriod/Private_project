from pathlib import Path
from torch import Tensor
import torchaudio

# 'parent_directory'에 'data'라는 하위 폴더를 추가

parent_directory = Path(__file__).parent

data_directory = parent_directory / 'data'

# 절대 경로를 얻기 위해 resolve() 메서드 사용

print(data_directory)

yesno_data = torchaudio.datasets.LIBRISPEECH(data_directory,url="train-clean-100" , download=True)