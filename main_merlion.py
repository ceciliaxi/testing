import argparse
import os
import json
from merlion_ad import MerlionBaseline


def main():
    train_arg_parser = argparse.ArgumentParser(description="parser")

    # Parameters for the dataset  
    train_arg_parser.add_argument("--dataset", type=str, default='SMD', help='dataset name')
    train_arg_parser.add_argument("--feat_num", type=int, default=38, help="number of features")
    train_arg_parser.add_argument("--data_path", type=str, default='data/SMD_DA/tr_1-1_2-1_3-2_te_3-7', 
                                  help='folder root of the data')
    train_arg_parser.add_argument("--model_name", type=str, default='VAE', 
                                  help="selected AD model")

    args = train_arg_parser.parse_args()

    setting = '{}_{}_{}_{}'.format(
        args.dataset,
        args.feat_num, 
        args.data_path.split('/')[-1],
        args.model_name) 
    print(setting)

    model_obj = MerlionBaseline(flags=args)
    train_time = model_obj.train()

    # after training, we should test the held out domain
    test_results, test_time = model_obj.test()

    # ######################## REVISED HERE ######################### 
    # Write the expected results to file
    results_output = ' & '.join([str(i) for i in 
                                    [# F1 with optimal threshold
                                        test_results['pak']['wo_pa']['basic']['f1'], 
                                        test_results['pak']['w_pa']['basic']['f1'], 
                                        test_results['pak_vk']['f1'], 
                                        # F1 with POT threshold
                                        test_results['pak_pot']['wo_pa']['basic']['f1'], 
                                        test_results['pak_pot']['w_pa']['basic']['f1'],
                                        test_results['pak_vk_pot']['f1'], 
                                        # AUCs
                                        test_results['pak']['auc']['roc_auc'], 
                                        test_results['pak']['auc']['pr_auc'],
                                        test_results['pak']['range_auc']['range_roc_auc'],
                                        test_results['pak']['range_auc']['range_pr_auc'], 
                                        # Running time
                                        train_time, test_time]]) 
    print(results_output)
    result_path = './test_results/' + setting + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    result_f = open(result_path + 'latex_table.txt', 'w')
    result_f.write(results_output + '\n')
    with open(result_path + 'full_results.txt', 'w') as convert_file:
        convert_file.write(json.dumps(str(test_results)))
    ################################################################

if __name__ == "__main__":
    main()
