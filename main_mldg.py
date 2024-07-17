import argparse
import os
import json
from model import ModelMLDG


def main():
    train_arg_parser = argparse.ArgumentParser(description="parser")

    # Parameters for the dataset  
    train_arg_parser.add_argument("--dataset", type=str, default='SMD', help='dataset name')
    train_arg_parser.add_argument("--data_root", type=str, default='data/SMD_DA/', 
                                  help='folder root of the data')
    train_arg_parser.add_argument("--feat_num", type=int, default=38, help="number of features")
    train_arg_parser.add_argument("--train_data", type=str, default=['tr_1-1','tr_2-1', 'tr_3-2'], 
                                  help="list of training domains.")  
    train_arg_parser.add_argument("--test_data", type=str, default='te_3-7', 
                                  help="unseen test domain")
    train_arg_parser.add_argument("--val_ratio", type=float, default='0.1', 
                                  help="ratio for validation splitted from training data")
    
    # Parameters for the model training
    train_arg_parser.add_argument("--sequence_len", type=int, default=50, #default=2, #default=1,
                                  help='sliding window size')
    train_arg_parser.add_argument("--inner_loops", type=int, default=500, #default=45001,
                                  help="loops to repeat the training process")
    train_arg_parser.add_argument("--batch_size", type=int, default=128, #default=1024, #default=64,
                                  help="batch size for training, default is 64")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3, #default=5e-4,
                                  help='learning rate of the model')
    train_arg_parser.add_argument("--step_size", type=int, default=15000,
                                  help="period of learning rate decay")
    train_arg_parser.add_argument("--meta_step_size", type=float, default=5e-1, #default=1e-1, #default=5e-1,
                                  help='meta step size')   
    train_arg_parser.add_argument("--meta_val_beta", type=float, default=1, 
                                  help='the strength of the meta val loss')

    # Parameters for logging the models
    train_arg_parser.add_argument("--logs", type=str, default='logs/',
                                  help='logs folder to write log')
    train_arg_parser.add_argument("--model_path", type=str, default='model/', 
                                  help='folder for saving model')
    train_arg_parser.add_argument("--state_dict", type=str, default='',
                                  help='model of pre trained')

    # Parameters for logging the results
    train_arg_parser.add_argument("--test_every", type=int, default=500, #default=500,
                                help="number of test every steps")
    train_arg_parser.add_argument("--debug", type=bool, default=False,
                                  help='whether for debug mode or not')

    args = train_arg_parser.parse_args()

    setting = '{}_{}_{}_vr{}_sl{}_il{}_bs{}_lr{}_ss{}_ms{}_mb{}'.format(
        args.dataset,
        '_'.join(args.train_data),
        args.test_data,
        args.val_ratio,
        args.sequence_len,
        args.inner_loops,
        args.batch_size,
        args.lr,
        args.step_size,
        args.meta_step_size,
        args.meta_val_beta)
    print(setting)

    model_obj = ModelMLDG(flags=args)
    train_time = model_obj.train(flags=args)

    # after training, we should test the held out domain
    test_results, test_time = model_obj.heldout_test(flags=args)

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
