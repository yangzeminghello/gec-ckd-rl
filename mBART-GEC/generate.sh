model_dir=/root/autodl-tmp/model/gec_de_1/checkpoint_best.pt # fix if you moved the checkpoint

fairseq-generate path_2_data \
  --path $model_dir/model.pt \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -t tgt -s src \
  --bpe 'sentencepiece' --sentencepiece-model /root/generic-pretrained-GEC-master/mBART-GEC/mbart.cc25/sentence.bpe.model \
  --sacrebleu --remove-bpe 'sentencepiece' \
  --batch-size 32 --langs de > de

cat en_ro | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > en_ro.hyp
cat en_ro | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > en_ro.ref
sacrebleu -tok 'none' -s 'none' en_ro.ref < en_ro.hyp