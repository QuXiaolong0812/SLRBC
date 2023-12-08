@REM python ./main.py	--model_type="lstm" --use_biword=0 --use_bert=0 --use_robert=0 --use_seftlexion=1 --log_file="log/ccks2019/ccks2019_SLRBC(char + SoftLexicon).log"
@REM python ./main.py	--model_type="lstm" --use_biword=1 --use_bert=0 --use_robert=0 --use_seftlexion=0 --log_file="log/ccks2019/ccks2019_SLRBC(char_bichar_lstm-crf).log"
@REM python ./main.py	--model_type="lstm" --use_biword=0 --use_bert=1 --use_robert=0 --use_seftlexion=0 --log_file="log/ccks2019/ccks2019_char-BERT-BiLSTM-CRF.log"
python ./main.py	--model_type="lstm" --use_biword=0 --use_bert=1 --use_robert=0 --use_seftlexion=1 --log_file="log/ccks2018/ccks2018_softlexicon_bert-bilstm-crf.log"


