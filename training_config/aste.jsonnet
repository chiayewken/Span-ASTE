local template = import "template.libsonnet";

template.SpanModel {
  bert_model: "bert-base-uncased",
  cuda_device: 0,
  data_paths: {
    train: "/tmp/train.json",
    validation: "/tmp/dev.json",
    test: "/tmp/test.json",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
  },
  model +: {
    modules +: {
      relation: {
        spans_per_word: 0.5,
        use_distance_embeds: false,
        use_ner_scores_for_prune: false,
        use_span_pair_aux_task: false,
        use_span_pair_aux_task_after_prune: false,
        use_pruning: true,
        use_pair_feature_multiply: false,
        use_pair_feature_maxpool: false,
        use_pair_feature_cls: false,
        use_span_loss_for_pruners: false,
        use_ope_down_project: false,
        use_pairwise_down_project: false,
        use_classify_mask_pruner: false,
        use_bi_affine_classifier: false,
        use_bi_affine_pruner: false,
        neg_class_weight: -1,
        use_focal_loss: false,
        focal_loss_gamma: 2,
        span_length_loss_weight_gamma: 0.0,
        use_bag_pair_scorer: false,
        use_bi_affine_v2: false,
        use_single_pool: false,
      },
      ner: {
        use_bi_affine: false,
        neg_class_weight: -1,
        use_focal_loss: false,
        focal_loss_gamma: 2,
        use_double_scorer: false,
        use_gold_for_train_prune_scores: false,
        use_single_pool: false,
      }
    },
    use_ner_embeds: false,
    span_extractor_type: "endpoint",
    relation_head_type: "new",
    use_span_width_embeds: true,
    use_bilstm_after_embedder: false,

    use_double_mix_embedder: false,
//    use_double_mix_embedder: true,
//    embedder +: {
//      token_embedders +: {
//        bert +: {
//          type: 'double_mix_ptm',
//        },
//      },
//    },

//    feature_size: 50,
  },
  target_task: "relation",
  trainer +: {
    num_epochs: 10,  # Set to < 5 for quick debugging
    optimizer: {
      type: 'adamw',
      lr: 1e-3,
      weight_decay: 0.0,
      parameter_groups: [
        [
          ['_matched_embedder'],  # May need to switch if using different embedder type in future
//          ['_embedder'],
          {
            lr: 5e-5,
            weight_decay: 0.01,
            finetune: true,
          },
        ],
        [
          ['scalar_parameters'],
          {
            lr: 1e-2,
          },
        ],
      ],
    }
  },
}
