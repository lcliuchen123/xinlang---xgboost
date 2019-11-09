
from predict_data import *
import threading
from cost_time import *
import time
import multiprocessing

# 协程、线程和进程的区别
# 多线程适用于I/O密集型任务，多进程适用于CPU密集型任务
@run_time
def multi_run():
    train_file_name = './data/weibo_train_data.txt'
    test_file_name = './data/weibo_predict_data.txt'
    cate_feature = ['weight_time', 'is_aite', 'is_url', 'is_theme', 'is_face']  # 分类变量列表
    y_columns_list = ['forward_count', 'comment_count', 'like_count']  # 因变量的列名列表
    column_list = ['uid', 'time']  # get_features需要处理的列名列表
    eval_func = mae_score
    predict_obj = Predict_data(train_file_name, cate_feature, column_list,
                               y_columns_list, eval_func, test_file_name)
    start_time = time.time()
    # predict_obj.create_features()
    predict_obj.get_all_result()
    end_time = time.time()
    print('the cost of time is %f' % (end_time - start_time))  # 54614s

if __name__ == "__main__":
    start_time = time.time()
    t = threading.Thread(target=multi_run) #多线程
    t.start()
    t.join()
    end_time = time.time()
    print("多线程消耗时间：",(end_time - start_time)) #29975.68s

    # mstart_time = time.time()
    # p=multiprocessing.Process(target=multi_run)#多进程
    # p.start()
    # p.join()
    # mend_time = time.time()
    # print('the cost of multiprocess time is %f' % (mend_time - mstart_time))
