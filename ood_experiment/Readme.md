You just need to have following files to run:
 - train_amazon.csv,test_amazon.csv,test_sst2.csv,test_yahoo.csv,youtube.csv
 - amazon_llora_teacher_probs .pt
You need to install xlsxwriter sir: `pip install xlsxwriter`

After having all these you just need to run:
 - `python amazon_dirichlet_student_ood.py --mode learnable` for ood with learnable alpha0
 - `python amazon_dirichlet_student_ood.py` for standard dirichlet with seed 2
