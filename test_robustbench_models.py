import torch
import robustbench
from robustbench.model_zoo.cifar10 import linf
from robustbench.utils import load_model
from robustbench.data import load_cifar10
from autoattack import AutoAttack
from robustbench.data import *
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_cifar10_train_data(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR10(root=data_dir,
                               train=True,
                               transform=transforms_test,
                               download=True)
    return _load_dataset(dataset, n_examples)



x_test, y_test = load_cifar10(n_examples=200)

x_train, y_train = load_cifar10_train_data(n_examples=200)


errors = []
models = []
names = []

large_models = {"Gowal2020Uncovering_70_16", "Gowal2020Uncovering_70_16_extra", "Gowal2020Uncovering_34_20", 
                   "Sehwag2021Proxy_ResNest152", "Rebuffi2021Fixing_106_16_cutmix_ddpm", "Rebuffi2021Fixing_70_16_cutmix_ddpm", 
                   "Rebuffi2021Fixing_70_16_cutmix_extra", "Gowal2021Improving_70_16_ddpm_100m", "Kang2021Stable", 
                   "Jia2022LAS-AT_70_16", "Pang2022Robustness_WRN70_16", "Debenedetti2022Light_XCiT-L12",
                   "Huang2022Revisiting_WRN-A4", "Wang2023Better_WRN-70-16", "Bai2023Improving_edm", "Peng2023Robust"}


model = load_model("Sehwag2021Proxy", dataset='cifar10', threat_model='Linf')

l2_eps_orig = 1.74 / 8
l_inf_eps_orig = 1 / 255

for i in range(1, 17, 1):
    print(l_inf_eps_orig * i * 0.5)

sort_dict = sorted(linf.items())[16:] #fallback 
with tqdm(total=len(sort_dict) - len(large_models)) as pbar:
    for key in sort_dict:
        key = key[0]
        if key in large_models:
            bs = 8
        else:
            bs = 500
        if key not in large_models:
            try:
                model = load_model(key, dataset='cifar10', threat_model='Linf')
                model = model.to(device)
                print()
                print()
                print(f"model is - {key}")
                # names.append(key)

                print(f"attack linf on test data")
                for i in range(1, 17, 1):
                    l_inf_eps = l_inf_eps_orig * i * .5
                    adversary = AutoAttack(model, norm='Linf', eps=l_inf_eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                    adversary.apgd.n_restarts = 1
                    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
                print()
                print()

                print(f"attack l2 on test data")
                for i in range(1, 17, 1):   
                    l2_eps = l2_eps_orig * i * .5
                    adversary = AutoAttack(model, norm='L2', eps=l2_eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                    adversary.apgd.n_restarts = 1
                    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
                print()
                print()

                print(f"attack linf on train data")  
                for i in range(1, 17, 1):
                    l_inf_eps = l_inf_eps_orig * i * .5 
                    adversary = AutoAttack(model, norm='Linf', eps=l_inf_eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                    adversary.apgd.n_restarts = 1
                    x_adv = adversary.run_standard_evaluation(x_train, y_train, bs=bs)
                print()
                print()

                print(f"attack l2 on train data") 
                for i in range(1, 17, 1):
                    l2_eps = l2_eps_orig * i * .5   
                    adversary = AutoAttack(model, norm='L2', eps=l2_eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                    adversary.apgd.n_restarts = 1
                    x_adv = adversary.run_standard_evaluation(x_train, y_train, bs=bs)
                print()
                print()

                # model.to("cpu")
            except Exception as e:
                # By this way we can know about the type of error occurring
                print("The error is: ",e)
                errors.append(key)
                print(f"key error - {key}")
            pbar.update(1)
    print(f"done l2 eps - {l2_eps}, linf eps - {l_inf_eps}")
print(f"done running on {len(linf)} models")




