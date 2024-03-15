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



l2_eps = 1.74
l_inf_eps = 8. / 255



sort_dict = sorted(linf.items())
with tqdm(total=len(sort_dict) - len(large_models)) as pbar:

    for key in sort_dict:

        key = key[0]
        model = load_model(key, dataset='cifar10', threat_model='Linf')
        model = model.to(device)
        print(f"\n\nmodel is - {key}")

        print(f"attack linf on test data")
        adversary = AutoAttack(model, norm='Linf', eps=l_inf_eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
        adversary.apgd.n_restarts = 1
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
        print(end="\n\n")


        print(f"attack l2 on test data")
        adversary = AutoAttack(model, norm='L2', eps=l2_eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
        adversary.apgd.n_restarts = 1
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
        print(end="\n\n")

        print(f"attack linf on train data")  
        adversary = AutoAttack(model, norm='Linf', eps=l_inf_eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
        adversary.apgd.n_restarts = 1
        x_adv = adversary.run_standard_evaluation(x_train, y_train, bs=bs)
        print(end="\n\n")

        print(f"attack l2 on train data") 
        adversary = AutoAttack(model, norm='L2', eps=l2_eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
        adversary.apgd.n_restarts = 1
        x_adv = adversary.run_standard_evaluation(x_train, y_train, bs=bs)
        print(end="\n\n")

        pbar.update(1)
    print(f"done l2 eps - {l2_eps}, linf eps - {l_inf_eps}")
print(f"done running on {len(linf)} models")




