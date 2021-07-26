#!/usr/bin/env bash
set -e

DEVICE=$1
PYTHON="python -m pdb -c continue"
echo DEVICE="$DEVICE"

rm -rf model*
mkdir -p models
$PYTHON aste/main.py \
  --names 14lap,14lap,14lap,14lap,14lap,14res,14res,14res,14res,14res,15res,15res,15res,15res,15res,16res,16res,16res,16res,16res \
  --seeds 0,1,12,123,1234,0,1,12,123,1234,0,1,12,123,1234,0,1,12,123,1234 \
  --trainer__cuda_device "$DEVICE" \
  --trainer__num_epochs 10 \
  --trainer__checkpointer__num_serialized_models_to_keep 1 \
  --model__span_extractor_type "endpoint" \
  --model__modules__relation__use_single_pool False \
  --model__relation_head_type "proper" \
  --model__use_span_width_embeds True \
  --model__modules__relation__use_distance_embeds True \
  --model__modules__relation__use_pair_feature_multiply False \
  --model__modules__relation__use_pair_feature_maxpool False \
  --model__modules__relation__use_pair_feature_cls False \
  --model__modules__relation__use_span_pair_aux_task False \
  --model__modules__relation__use_span_loss_for_pruners False \
  --model__loss_weights__ner 1.0 \
  --model__modules__relation__spans_per_word 0.5 \
  --model__modules__relation__neg_class_weight -1

# Glove + BiLSTM Flags
#  --dataset_reader__token_indexers "{tokens: {type: 'single_id', lowercase_tokens: True}}" \
#  --model__embedder__token_embedders "{tokens: {type: 'embedding', embedding_dim: 300, pretrained_file: 'https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz', trainable: True}}" \
#  --model__use_bilstm_after_embedder True \

# New flags (these are default values)
# Proposed changes:
# Set "span_width_embeds" and "use_pair_multiply" to False
# Change "span_extractor_type" to "bi_affine"  (flag above)
# Try different "neg_class_weight" to eg 0.1,0.5 (-1 means None)
# Flip other boolean flags below
# Try different "focal_loss_gamma" eg 2,3,4
#
#  --model__modules__relation__use_classify_mask_pruner False \
#  --model__modules__relation__use_bi_affine_classifier False \
#  --model__modules__relation__use_bi_affine_pruner False \
#  --model__modules__relation__neg_class_weight -1 \
#  --model__modules__relation__use_focal_loss False \
#  --model__modules__relation__focal_loss_gamma 2 \
#  --model__modules__ner__use_bi_affine False \
#  --model__modules__ner__neg_class_weight -1 \
#  --model__modules__ner__use_focal_loss False \
#  --model__modules__ner__focal_loss_gamma 2 \

# Even more baseline flags:
# --model__modules__relation__use_pruning False \
# --model__use_new_relation_head False \

# Default arguments above is a demo of baseline config
# They can be mixed-and-matched
# ner: Set to 0.0 if use_span_loss_for_pruners==True
# use_pair_feature_multiply: set False if use_distance_embeds or use_pair_feature_maxpool
