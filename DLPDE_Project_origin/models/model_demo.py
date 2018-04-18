import thulac
import kenlm
# thu1 = thulac.thulac(seg_only=False,user_dict='/home/peng.qiu/RNNLM_Project/configs/user_dict.txt')
import gensim
from gensim.models import Word2Vec
from data_loader.data_loader import *
from models.RNNLM_Model_bi import *
from models.RNNLM_Model import *
from trainers.LMTrainer import *
from utils.config import *
from utils.dirs import *
from utils.utils import get_args
from utils.logger import Logger
from models.test_model import *
import re
def load_n_gram_model():
    lm = kenlm.LanguageModel('/data2/pengqiu/mix6_10.lm')
    return lm
#to do :短句跳过，停用词构建，分词修正，单字修改(变充电边播放), 
class correction_model():
    def __init__(self):
        """load数据"""
        # self.config = config
        self.sim_dict = pickle.load(open('/home/peng.qiu/nlc-master/dataset/simp_simplified.pickle', 'rb'))
        _,self.vocab_to_int = get_config_from_json('configs/vocab_to_int.json')
        self.w2v_model = Word2Vec.load('/data2/pengqiu/LM_data/w2v_news_size150.bin')
        # self.w2v_vocab = self.w2v_model.wv.vocab
        self.w2v_vocab = {}
        #出现次数少于50的都忽略
        for item,value in self.w2v_model.wv.vocab.items():
            if value.count>40:
                self.w2v_vocab[item] = value
                
        self.thu = thulac.thulac(seg_only=False,user_dict='/data2/pengqiu/LM_data/cut_dict_2.txt')
        self.first_name = [i.strip() for i in open('/home/peng.qiu/RNNLM_Project/configs/firstname.txt','r').readlines()]
        # self.user_dict = [i.strip() for i in open('/data2/pengqiu/LM_data/cut_dict_2.txt').readlines()]
        self.hot_list = [i.strip() for i in open('/data2/pengqiu/LM_data/singer_dict.txt').readlines()]
        self.casual_word = [i.strip() for i in open('/home/peng.qiu/RNNLM_Project/casual_word.txt').readlines()]
        # self.lm = load_n_gram_model()
        self.lm = kenlm.LanguageModel('/data2/pengqiu/mix6_10.lm')
    def candidate_word(self,word):
        """得到可能的词，包括少一个字，乱序， 差一个字相似读音， 差两个字相似读音, 补上一个字"""
        try:
            letters    = self.vocab_to_int.keys()
            splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
            deletes    = [L + R[1:]               for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
            replaces   = [L + c + R[1:]           for L, R in splits if R for c in self.sim_dict[R[0]]]
            replaces_2   = [L + c + d + R[2:]     for L, R in splits if len(R)>1 for c in self.sim_dict[R[0]] for d in self.sim_dict[R[1]]]
            inserts    = [L + c + R               for L, R in splits for c in letters]
            #     return set(deletes + transposes + replaces + inserts)
            deletes    = [i                       for i in deletes if len(i)>1]
            #去掉只有一个字的备选词
        except:
            #print('出现少见词')
            return set()
        return set(deletes + transposes + replaces + replaces_2 + inserts)
    
    
    def correction(self,sentence):
        """处理句子"""
        message =  self.dispose_sentence(sentence)
        (sentence,sentence_cut,words_lr2,oov) = message
        #print(message)
        init_w2v_prob = self.w2v_model.score([sentence_cut])[0]
        # init_n_gram_prob = self.lm.score(self.to_lm_type(sentence))
        if self.init_score >-12:
            #print('句子正确')
            return sentence
        if words_lr2==[]:
            #处理只有一个字的词的句子
            #print('只有一个字的词的句子')
            return self.all_one_word(message)
        #不在词库中的词
        if not oov:
            #当没有oov的词
            #print('no oov')
            return self.deal_no_oov(message)
        else:
            #当有oov的词
            #print('oov',oov)
            return self.deal_oov(message)
        
    def dispose_sentence(self,sentence):
        #清洗数据
        sentence = re.sub(u'[^\u4e00-\u9fa5]+',"",sentence)
        # sentence_cut = self.thu.cut(sentence)
        # sentence_cut_lm = ''.join([j[0]+' ' for j in sentence_cut])
        # self.init_score = self.lm.score(sentence_cut_lm)
        self.init_score, sentence_cut = self.cut_and_score(sentence)
        sentence_cut = [i for i in sentence_cut if len(i)==1 or i not in self.casual_word]
        #切割后的句子，如：['我', '要', '听', '歌曲', '周杰轮', '的', '听妈妈的话'] 并去掉常见字词
        words_lr2 = [i for i in sentence_cut if len(i)>1]
        #两个字以上的词
        oov = [i for i in sentence_cut if i not in self.w2v_vocab]
        # and i not in self.user_dict 
        message = (sentence,sentence_cut,words_lr2,oov)
        #把所有信息组合用于函数间的传输
        return message
    
    def cut(self,sentence):
        """修正后的分词"""
        sentence_tmp = self.thu.cut(sentence)
        sentence_cut = ''
        for i in range(len(sentence_tmp)):
            word = sentence_tmp[i][0]
            if i<len(sentence_tmp)-1 and len(word)==1 and word in self.first_name and len(sentence_tmp[i+1][0])<3:
                #对姓氏进行修正,不在最后一个，长度为1，为常见姓氏，下一词大小小于3
                sentence_cut += word
            elif '放' == word[0] and len(word)==2 and word[1] in self.first_name:
                #对放字进行修正
                sentence_cut += word[0] + ' ' + word[1]
            elif '的' == word[0] and len(word)==2:
                sentence_cut += word[0] + ' ' + word[1]+ ' '
            else:
                sentence_cut += word + ' '
        sentence_cut = sentence_cut.strip().split(' ')
        return sentence_cut 

    def cut_and_score(self,sentence):
        """对一句话进行分词并且计算得分"""
        sentence_cut = self.cut(sentence)
        sentence_cut_lm = ''.join([j+' ' for j in sentence_cut])
        #将分词结果转换为n-gram模型可以使用的结果
        score = self.lm.score(sentence_cut_lm)
        return score,sentence_cut 
    
    def deal_oov(self,message):
        """处理有oov的情况，当所有的组合的最大得分大于0.5时，改正句子，返回正确的句子"""
        (sentence,sentence_cut,words_lr2,oov) = message
        note = True
        result = []
        for oov_i in oov:
            other_words = [w for w in words_lr2 if w not in oov]
            # other_words = [w for w in sentence_cut if w not in oov]
            #除了oov以外的所有词                
            cand_words = self.candidate_word(oov_i)
            hot = self.in_hot_list(oov_i,cand_words)
            cand_words = [w for w in cand_words if w in self.w2v_vocab]
            cand_words = [cand_word for cand_word in cand_words if cand_word in self.w2v_vocab]
            # self.w2v_vocab[cand_word].count>100
            if hot !=0:
                #有热门词
                note = False
                #print('有热门词')
                sentence = sentence.replace(oov_i,hot)
                continue
            if cand_words == []:
                #print(oov_i,'无备选词')
                #!!!对前后的词合并，且生成备选词
                continue
            #print(cand_words)
            #备选词
            sentences_new,Score = self.score_cand_words(message,oov_i,cand_words,other_words)
            Score_max = np.max(Score)
            replace_word = cand_words[np.argmax(Score)]
            if Score_max>0.3:
                note = False
                sentence = sentence.replace(oov_i,replace_word)
        if note:
            #print('不进行纠正')
            #对前后词进行合并
            return sentence
        else:
            #print('进行纠正')
            return sentence 
    def merge_near(self,message,oov_i):
        """合并附近的词"""
        #未完成
        (sentence,sentence_cut,words_lr2,oov) = message
        place = sentence.find(oov_i)
        oov_new = sentence[place-1]+oov_i
        place_2 = sentence_cut.find(oov_i)
        cand_2 = self.replace_one_word(oov_i+sentence[place+1])
        
    def score_cand_words(self,message,oov_i,cand_words,other_words):
        """根据备选词跟其他词还有原始句子，综合得到得分"""
        (sentence,sentence_cut,words_lr2,oov) = message
        cand_words_rate = [self.word_rate(oov_i,cand_word) for cand_word in cand_words]
        cand_count = np.asarray([self.w2v_vocab[cand_word].count for cand_word in cand_words])
        count_score = cand_count/np.sum(cand_count)
        #根据出现次数计算得分
        sentences_new = [sentence.replace(oov_i,cand_word) for cand_word in cand_words]
        #所有可能新句子
        lm_score = np.asarray([self.cut_and_score(sentence_new)[0] for sentence_new in sentences_new])
        #语言模型得分
        if len(cand_words)==1:
            if cand_count[0]>80 and lm_score[0]>self.init_score and len(cand_words[0]) == len(oov_i):
                return sentences_new,[1]
            else:
                return sentences_new,[0]
        lm_score_tmp = lm_score - np.sum(lm_score)
        lm_score_tmp = lm_score_tmp / np.sum(lm_score_tmp)
        #正则化
        if other_words!=[]:
            w2v_score = np.asarray([self.max_prob(cand_word,other_words)[0] for cand_word in cand_words])
            #w2v 得分
            Score = (count_score+lm_score_tmp+w2v_score)/3
        else:
            Score = (count_score+lm_score_tmp)/2
        Score = Score *np.asarray(cand_words_rate)
        return sentences_new,Score
    
    def word_rate(self,word,cand_word):
        """对备选词关于原始词计算得分"""
        rate = 1
        if cand_word in self.hot_list:
            #出现热门词，rate乘以2
            rate = rate *2
        if len(cand_word)!=len(word):
            rate = rate *0.5
        else: 
            count =0
            for i in range(len(word)):
                if cand_word[i]!=word[i]:
                    count +=1
            rate = rate * (0.8**count)
        return rate
        
        
    def deal_no_oov(self,message):
        """处理没有oov的情况，当所有的组合的最大得分大于0.5时，改正句子，返回正确的句子"""
        (sentence,sentence_cut,words_lr2,oov) = message
        note = True
        result = []
        for word in words_lr2:
            #每个字数大于2的词
            other_words = [w for w in words_lr2 if w!=word]
            if other_words ==[]:
                #如果没有其他词,说明只有一个词
                #print('只有一个lr2词')
                return self.all_one_word(message)
            #其他词
            cand_words = self.candidate_word(word)
            cand_words = [w for w in cand_words if w in self.w2v_vocab]
            cand_words.append(word)
            #备选词
            if cand_words == []:
                # #print(oov_i,'无备选词')
                #!!!对前后的词合并，且生成备选词
                continue
            #print(cand_words)
            sentences_new,Score = self.score_cand_words(message,word,cand_words,other_words)
            Score_max = np.max(Score)
            replace_word = cand_words[np.argmax(Score)]
            if Score_max>0.3:
                note = False
                sentence = sentence.replace(word,replace_word)
        if note:
            #print('不进行纠正')
            #对前后词进行合并
            return sentence
        else:
            #print('进行纠正')
            return sentence 
    def replace_one_word(self,message):
        (sentence,sentence_cut,words_lr2,oov) = message
        cand_replaces = []
        for s in sentence_cut:
            if len(s) ==1:
                replaces = [sentence.replace(s,c) for c in self.sim_dict[s]]
                cand_replaces.extend(replaces)
        return cand_replaces
    # def replace_one_word(self,word):
    #     """输入一个词，把每个词进行替换，生成所有可能的结果，/去除不在词表的数据/"""
    #     replaces = [word.replace(w,c) for w in word for c in self.sim_dict[w]]
    #     # replaces = [i for i in replaces if i in self.w2v_vocab]
    #     return replaces                
                

    def all_one_word(self,message):
        """对只有一个字的词的句子进行处理"""
        (sentence,sentence_cut,words_lr2,oov) = message
        # length = len(sentence_cut)
        # W = ''.join(sentence_cut)
        W = sentence
        cand_sentences = self.replace_one_word(message)
        #只有出现300以上的才可以作为备选词
        sentences_score = [self.cut_and_score(cand_sentence) for cand_sentence in cand_sentences]
        if sentences_score == []:
            #print('无备选可能，认为是正确的')
            return sentence
        max_score,max_sentence = max(sentences_score)
        R = ''.join([j for j in max_sentence])
        #替换后的句子，但是可能有常用词已经被去除
        sentence_new = sentence.replace(W,R)
        if max_score>self.init_score:
            #替换后得分变高
            #print('将句子替换为',sentence_new,'原来lm得分:',self.init_score,'修改后lm得分',max_score)
            return sentence_new
        else:
            #print('替换后得分没有提高，不进行替换')
            #print(sentence_new)
            return sentence
        
    def max_prob(self,word,other_words):
        """返回跟目标词最相关的词以及相似度"""
        S = [self.w2v_model.similarity(word,other_word) for other_word in other_words]
        max_s = max(S)
        max_w = other_words[np.argmax(S)]
        return max_s,word,max_w    
    
    def in_hot_list(self,word,cand_words):
        """判断备选集中是否有热门词"""
        h =[]
        for cand_word in cand_words:
            if cand_word in self.hot_list:
                h.append(cand_word)
        if len(h)==1 and len(h[0])==len(word):
            #只有一个热门词且大小相同才会替换
            return h[0]
        else:
            return 0
                