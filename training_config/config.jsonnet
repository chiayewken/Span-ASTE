{
  "data_loader": {
    "sampler": {
      "type": "random"
    }
  },
  "dataset_reader": {
    "max_span_width": 8,
    "token_indexers": {
      "bert": {
        "max_length": 512,
        "model_name": "bert-base-uncased",
        "type": "pretrained_transformer_mismatched"
      }
    },
    "type": "span_model"
  },
  "model": {
    "embedder": {
      "token_embedders": {
        "bert": {
          "max_length": 512,
          "model_name": "bert-base-uncased",
          "type": "pretrained_transformer_mismatched"
        }
      }
    },
    "feature_size": 20,
    "feedforward_params": {
      "dropout": 0.4,
      "hidden_dims": 150,
      "num_layers": 2
    },
    "initializer": {
      "regexes": [
        [
          "_span_width_embedding.weight",
          {
            "type": "xavier_normal"
          }
        ]
      ]
    },
    "loss_weights": {
      "ner": 1.0,
      "relation": 1
    },
    "max_span_width": 8,
    "module_initializer": {
      "regexes": [
        [
          ".*weight",
          {
            "type": "xavier_normal"
          }
        ],
        [
          ".*weight_matrix",
          {
            "type": "xavier_normal"
          }
        ]
      ]
    },
    "modules": {
      "ner": {
        "focal_loss_gamma": 2,
        "neg_class_weight": -1,
        "use_bi_affine": false,
        "use_double_scorer": false,
        "use_focal_loss": false,
        "use_gold_for_train_prune_scores": false,
        "use_single_pool": false
      },
      "relation": {
        "focal_loss_gamma": 2,
        "neg_class_weight": -1,
        "span_length_loss_weight_gamma": 0,
        "spans_per_word": 0.5,
        "use_bag_pair_scorer": false,
        "use_bi_affine_classifier": false,
        "use_bi_affine_pruner": false,
        "use_bi_affine_v2": false,
        "use_classify_mask_pruner": false,
        "use_distance_embeds": true,
        "use_focal_loss": false,
        "use_ner_scores_for_prune": false,
        "use_ope_down_project": false,
        "use_pair_feature_cls": false,
        "use_pair_feature_maxpool": false,
        "use_pair_feature_multiply": false,
        "use_pairwise_down_project": false,
        "use_pruning": true,
        "use_single_pool": false,
        "use_span_loss_for_pruners": false,
        "use_span_pair_aux_task": false,
        "use_span_pair_aux_task_after_prune": false
      }
    },
    "relation_head_type": "proper",
    "span_extractor_type": "endpoint",
    "target_task": "relation",
    "type": "span_model",
    "use_bilstm_after_embedder": false,
    "use_double_mix_embedder": false,
    "use_ner_embeds": false,
    "use_span_width_embeds": true
  },
  "trainer": {
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    "cuda_device": 0,
    "grad_norm": 5,
    "learning_rate_scheduler": {
      "type": "slanted_triangular"
    },
    "num_epochs": 10,
    "optimizer": {
      "lr": 0.001,
      "parameter_groups": [
        [
          [
            "_matched_embedder"
          ],
          {
            "finetune": true,
            "lr": 5e-05,
            "weight_decay": 0.01
          }
        ],
        [
          [
            "scalar_parameters"
          ],
          {
            "lr": 0.01
          }
        ]
      ],
      "type": "adamw",
      "weight_decay": 0
    },
    "validation_metric": "+MEAN__relation_f1"
  },
  "numpy_seed": 0,
  "pytorch_seed": 0,
  "random_seed": 0,
  "test_data_path": "",
  "train_data_path": "",
  "validation_data_path": ""
}
