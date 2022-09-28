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
      },
      "relation": {
        "spans_per_word": 0.5,
        "use_distance_embeds": true,
        "use_pruning": true,
      }
    },
    "span_extractor_type": "endpoint",
    "target_task": "relation",
    "type": "span_model",
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
