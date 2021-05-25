# Dent: Dynamic Defenses against Adversarial Attacks

This is the official project repository for [Fighting Gradients with Gradients: Dynamic Defenses against Adversarial Attacks](https://arxiv.org/abs/2105.08714) by
Dequan Wang, An Ju, Evan Shelhamer, David Wagner, and Trevor Darrell.

## Abstract

Adversarial attacks optimize against models to defeat defenses.
We argue that models should fight back, and optimize their defenses against attacks at test-time.
Existing defenses are static, and stay the same once trained, even while attacks change.
We propose a dynamic defense, defensive entropy minimization (dent), to adapt the model and input during testing by gradient optimization.
Our dynamic defense adapts fully at test-time, without altering training, which makes it compatible with existing models and standard defenses.
Dent improves robustness to attack by 20+ points (absolute) for state-of-the-art static defenses against AutoAttack on CIFAR-10 at epsilon (L_infinity) = 8/255.

## Example: Model Adaptation for Defense against AutoAttack on CIFAR-10

This example compares state-of-the-art adversarial training defenses, which are static, with and without our method for defense entropy minimization (dent), which is dynamic.
We evaluate against white-box and black-box attacks and report the worst-case accuracy across attack types.

- **Attacks**: [Standard AutoAttack](https://github.com/fra31/auto-attack), including untargeted [APGD-CE](https://arxiv.org/abs/2003.01690) (no restarts), targeted [APGD-DLR](https://arxiv.org/abs/2003.01690) (9 target classes), targeted [FAB](https://arxiv.org/abs/1907.02044) (9 target classes), black-box [Square](https://arxiv.org/abs/1912.00049) (5000 queries).
- **Static Defenses**: [RobustBench Model Zoo](https://github.com/RobustBench/robustbench#linf), including [Wu2020Adversarial_extra](https://arxiv.org/abs/2004.05884), [Carmon2019Unlabeled](https://arxiv.org/abs/1905.13736), [Sehwag2020Hydra](https://arxiv.org/abs/2002.10509), [Wang2020Improving](https://openreview.net/forum?id=rklOg6EFwS), [Hendrycks2019Using](https://arxiv.org/abs/1901.09960), [Wong2020Fast](https://arxiv.org/abs/2001.03994), [Ding2020MMA](https://openreview.net/forum?id=HkeryxBtPB).
- **Dent**: For this example, dent updates by test-time model adaptation, with sample-wise modulation of the parameters, but without input adaptation. Please see the paper for variants of model adaptation and input adaptation.

**Result**:

Dent improves adversarial/robust accuracy (%) by more than 30 percent (relative) against AutoAttack on CIFAR-10 while preserving natural/clean accuracy.
Our dynamic defense brings adversarial accuracy within 90\% of natural accuracy for the three most accurate methods tested (Wu 2020, Carmon 2019, Sehwag 2020).
The static defenses alter training, while dent alters testing, and so this separation of concerns makes dent compatible with many existing models and defenses.

| Model ID                | Paper                                                                                                                           | Natural (static) | Natural (dent) | Adversarial (static) | Adversarial (dent) | Venue        |
| ----------------------- | ------------------------------------------------------------                                                                    | ---------------- | -------------- | -------------------- | ------------------ | ------------ |
| Wu2020Adversarial_extra | [Adversarial Weight Perturbation Helps Robust Generalization](https://arxiv.org/abs/2004.05884)                                 | 88.25            | 87.65          | 60.04                | 80.33              | NeurIPS 2020 |
| Carmon2019Unlabeled     | [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/abs/1905.13736)                                              | 89.69            | 89.32          | 59.53                | 82.28              | NeurIPS 2019 |
| Sehwag2020Hydra         | [HYDRA: Pruning Adversarially Robust Neural Networks](https://arxiv.org/abs/2002.10509)                                         | 88.98            | 88.60          | 57.14                | 78.09              | NeurIPS 2020 |
| Wang2020Improving       | [Improving Adversarial Robustness Requires Revisiting Misclassified Examples](https://openreview.net/forum?id=rklOg6EFwS)       | 87.50            | 86.32          | 56.29                | 77.31              | ICLR 2020    |
| Hendrycks2019Using      | [Using Pre-Training Can Improve Model Robustness and Uncertainty](https://arxiv.org/abs/1901.09960)                             | 87.11            | 87.04          | 54.92                | 79.62              | ICML 2019    |
| Wong2020Fast            | [Fast is better than free: Revisiting adversarial training](https://arxiv.org/abs/2001.03994)                                   | 83.34            | 82.34          | 43.21                | 71.82              | ICLR 2020    |
| Ding2020MMA             | [MMA Training: Direct Input Space Margin Maximization through Adversarial Training](https://openreview.net/forum?id=HkeryxBtPB) | 84.36            | 84.68          | 41.44                | 64.35              | ICLR 2020    |

**Usage**:

```python
python cifar10a.py --cfg cfgs/dent.yaml MODEL.ARCH Wu2020Adversarial_extra
python cifar10a.py --cfg cfgs/dent.yaml MODEL.ARCH Carmon2019Unlabeled
python cifar10a.py --cfg cfgs/dent.yaml MODEL.ARCH Sehwag2020Hydra
python cifar10a.py --cfg cfgs/dent.yaml MODEL.ARCH Wang2020Improving
python cifar10a.py --cfg cfgs/dent.yaml MODEL.ARCH Hendrycks2019Using
python cifar10a.py --cfg cfgs/dent.yaml MODEL.ARCH Wong2020Fast
python cifar10a.py --cfg cfgs/dent.yaml MODEL.ARCH Ding2020MMA
```

## Correspondence

Please contact Dequan Wang, An Ju, and Evan Shelhamer at dqwang AT eecs.berkeley.edu, an_ju AT berkeley.edu, and shelhamer AT google.com.

## Citation

If the dent method or dynamic defense setting are helpful in your research, please consider citing our paper:

```bibtex
@article{wang2021fighting,
  title={Fighting Gradients with Gradients: Dynamic Defenses against Adversarial Attacks},
  author={Wang, Dequan and Ju, An and Shelhamer, Evan and Wagner, David and Darrell, Trevor},
  journal={arXiv preprint arXiv:2105.08714},
  year={2021}
}
```

Note: a [workshop edition of this project](https://aisecure-workshop.github.io/aml-iclr2021/papers/44.pdf) was presented at the ICLR'21 Workshop on Security and Safety in Machine Learning Systems.
