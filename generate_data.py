import pandas as pd
import numpy as np
np.random.seed(42)
data = {
    "student_id": range(1, 101),
    "study_hours": np.random.randint(1, 8, 100),
    "pre_test": np.random.randint(40, 70, 100),
    "post_test": np.random.randint(50, 90, 100),
    "attention_span": np.random.randint(20, 60, 100),
    "learning_method": np.random.choice(["visual", "auditory", "kinesthetic"], 100),
    "distractions": np.random.choice(["none", "music", "phone"], 100)
}
df = pd.DataFrame(data)
df.to_csv("student_behaviour.csv", index=False)
print("Dataset created successfully!")