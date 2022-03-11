# Noisy Label Learning for Security Defects

Software vulnerability prediction models learn from past vulnerability data to predict vulnerabilities in future code modules. However, we observe that it is infeasible to obtain a noise-free security defect dataset in practice. The non-vulnerable modules are difficult to be verified and determined as truly exploit free given the limited manual efforts available.

To address this issue, we propose novel learning methods that are robust to label impurities and can leverage the most from limited label data; noisy label learning. In this codebase, we investigate various noisy label learning methods applied to software vulnerability prediction.  

This repository contains vulnerability data for the Mozilla Firefox project from release 63-84, and the code for producing a software vulnerability prediction pipeline using this data. Features are extracted from the labeled file source code data, and then prediction models are trained on this data. We measure potential performance improvement from applying noisy label learning techniques onto this pipeline.

We show that noisy label learning techniques are effective for improving the capabilities of software vulnerability prediction. We observe that AUC and recall of baselines increases by up to 8.9% and 23.4%, respectively.  

## Running
Download the source code data from: https://drive.google.com/file/d/1bnUXGOaLr6E4cwgSmqr4K37aQe7PiUHq/view?usp=sharing and include it in the `data/mozilla/` folder.  
Download the software metric data from: https://drive.google.com/file/d/1E2BxK5ZVorYp3tuihYSTO9OnhAtIk0p6/view?usp=sharing and include it in the `data/` folder.  
Install dependencies using:
```
pip install requirements.txt
```  
Cleanlab requires custom installation. A label_flipping parameter has been added to the fit function, which controls whether noisy labels are flipped or pruned.
```
pip install -e code/svp/cleanlab/
```
Run source code from the root folder. Run all experiments using:
```
scripts/run.sh
```

## Citation
If using this work, please provide appropriate citation:  
```
@article{croft2021investigation,
  title={Noisy Label Learning for Security Defects},
  author={Croft, Roland and Babar, M. Ali and Chen, Huaming},
  journal={arXiv preprint arXiv:2203.04468},
  year={2022}
}
```

## Acknowledgments
“Copyright © Cyber Security Research Centre Limited 2021. This work has been supported by the Cyber Security Research Centre (CSCRC) Limited whose activities are partially funded by the Australian Government’s Cooperative Research Centres Programme. We are currently tracking the impact CSCRC funded research. If you have used this code in your project, please contact us at contact@cybersecuritycrc.org.au to let us know.”
