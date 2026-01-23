from xcpetion import build_xception_backbone
from models import get_xception_based_model
from utils import get_nof_params

# יצירת המודל (עדיין עם ה-FC המקורי של ImageNet כדי לבדוק את ה-Default)

original_xception = build_xception_backbone(pretrained=True) 
params_count = get_nof_params(original_xception)
print(f"Total parameters in default Xception: {params_count}")

# בדיקה למודל החדש שלך (עם ה-Head שהוספת)
my_model = get_xception_based_model()
new_params_count = get_nof_params(my_model)
print(f"Total parameters in MY Xception model: {new_params_count-params_count}")