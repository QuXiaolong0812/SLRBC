import json
import random
def GetData(file_path):
    '''
    从json文件中获取数据，返回数据的格式为[json1 , json2 , json2]是一个json组成的lits
    :param file_path:
    :return:
    '''
    papers = []
    with open(file_path,'r',encoding='utf-8-sig')as file:
        for line_num,line in enumerate(file.readlines()):
            try:
                dic = json.loads(line)
                papers.append(dic)
            except ValueError:
                print(">>>>File :"+ file_path+"   Error line:"+str(line_num)+"   >>>"+repr(line)+"<<<") #打印不是json的行
    return papers

def ProcessData(data,output_path):
    '''
    将data里面的json数据转换成BIO格式的标注数据，并保存的output_path中
    :param data: 格式为[json1 , json2 , json2]是一个json组成的lits
    :param output_path:
    :return:无
    '''
    print(">>>>output file:"+ output_path + "   data len:" + str(len(data)))
    with open(output_path,'w',encoding='utf-8') as ff:
        for line_dict in data:
            origin_text = line_dict['originalText']
            entities = line_dict['entities']
            label_text = ['O' for _ in range(len(origin_text))]
            for entity in entities:
                label_type = entity['label_type']
                start_pos = entity['start_pos']
                end_pos = entity['end_pos']
                label_text[start_pos] = 'B-' + label_type
                for j in range(start_pos + 1, end_pos):
                    label_text[j] = 'I-' + label_type
            for ii in range(len(origin_text)):
                ff.write(origin_text[ii] + ' ' + str(label_text[ii]) + '\n')
                if (origin_text[ii] in '!?。;'):
                    ff.write('\n')
    return

if __name__ == '__main__':
    # train_data = GetData('./data/yidu-s4k/subtask1_training_part1.txt')
    # print(len(train_data))
    # train_data2 = GetData('./data/yidu-s4k/subtask1_training_part2.txt')
    # print(len(train_data2))
    train_data3 = GetData('D:\code\python\Chinese-clinical-NER-master\data\subtask1_training_afterrevise.txt')
    print(len(train_data3))
    random.shuffle(train_data3) #将原始数据集打乱

    #奖数据分割成0---dev_start_index---test_start_index---end 一共分成3段
    dev_start_index = int(len(train_data3) * 0.8)
    test_start_index = int(len(train_data3) * 0.9)

    ProcessData(train_data3[0 : dev_start_index],"./data/yidu-s4k/train-bio.txt")
    ProcessData(train_data3[dev_start_index : test_start_index],"./data/yidu-s4k/dev-bio.txt")
    ProcessData(train_data3[test_start_index :],"./data/yidu-s4k/test-bio.txt")
