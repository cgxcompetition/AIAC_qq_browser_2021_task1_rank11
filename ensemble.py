import json

import numpy as np
from sklearn import decomposition
from zipfile import ZIP_DEFLATED, ZipFile


with open("output/csj_result_mae.json", "r") as load_f:
    json1 = json.load(load_f)
for key, item in json1.items():
    print(key)
    print(item)
    break

with open("output/csj_result_mse.json", "r") as load_f:
    json2 = json.load(load_f)
for key, item in json2.items():
    print(key)
    print(item)
    break

with open("output/gpj_result.json", "r") as load_f:
    json3 = json.load(load_f)
for key, item in json3.items():
    print(key)
    print(item)
    break

with open("output/xc_result.json", "r") as load_f:
    json4 = json.load(load_f)
for key, item in json4.items():
    print(key)
    print(item)
    break



def l2(x):
    y = np.linalg.norm(x, axis=0, keepdims=True)
    z = x / y
    return z


concat_result = {}
for key, value in json1.items():
    concat_result[key] = (
        np.concatenate(
            [
                np.array(l2(value)),
                np.array(l2(json2[key])),
                np.array(l2(json3[key])),
                np.array(l2(json4[key])),
            ], axis=0,
        )
    ).tolist()

concat_svd_result = {}
result_key = []
result_value = []

for key, value in concat_result.items():
    result_key.append(key)
    result_value.append(value)
print(len(result_key))
print(len(result_value))

estimator = decomposition.TruncatedSVD(
    n_components=256,
    algorithm="arpack",
)
svd_X_train = estimator.fit_transform(np.array(result_value))
for i in range(svd_X_train.shape[0]):
    concat_svd_result[result_key[i]] = svd_X_train[i].tolist()

with open("output/result.json", "w") as f:
    json.dump(concat_svd_result, f)

with ZipFile("output/result.zip", 'w', compression=ZIP_DEFLATED) as zip_file:
    zip_file.write("output/result.json")


# 不加L2正则的融合
# result = {}
# for key, value in json1.items():
#     result[key] = ((np.array(value) + np.array(json2[key]) + np.array(json3[key]) + np.array(json4[key]) + np.array(
#         json5[key])) / 5).tolist()
#
# for key, item in result.items():
#     print(key)
#     print(item)
#     break
#
# with open("result.json", "w") as f:
#     json.dump(result, f)

# 加L2正则的融合
# result = {}
# for key, value in json1.items():
#     result[key] = ((np.array(l2(value)) + np.array(l2(json2[key])) + np.array(l2(json3[key])) + np.array(
#         l2(json4[key])) + np.array(l2(json5[key]))) / 5).tolist()
#
# for key, item in result.items():
#     print(key)
#     print(item)
#     break
#
# with open("result_l2.json", "w") as f:
#     json.dump(result, f)
