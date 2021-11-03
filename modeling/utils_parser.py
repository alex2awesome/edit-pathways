def attach_model_arguments(parser):
    ## Required parameters
    parser.add_argument("--train_data_file_s3", default='data/news-discourse-training-data.csv', type=str)
    parser.add_argument("--eval_data_file_s3", default=None, type=str)
    parser.add_argument("--pretrained_files_s3", default='elmo', type=str)
    parser.add_argument("--finetuned_lm_file", default=None, type=str)
    parser.add_argument("--model_type", help="Which pretrained model to use - RoBERTa, BERT or GPT2", type=str)
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--env', type=str, default='local')
    parser.add_argument('--num_dataloader_cpus', type=int, default=10, help='Number of CPUs to use to process data')
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use per node.")
    parser.add_argument('--num_nodes', type=int, default=1, help="Number of nodes to use.")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    parser.add_argument('--experiment', type=str, default='baseline_non-sequential', help="Which experiment to run.")
    parser.add_argument('--random_split', action='store_true')
    parser.add_argument('--train_perc', type=float, default=1.0)
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--log_all_metrics', action='store_true')
    parser.add_argument('--tb_logdir', default=None, type=str, help="Path for tensorboard logs.")
    parser.add_argument('--use_deepspeed', action='store_true')

    ## optimization params
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=float, default=0, help="Num warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Whether to have a LR decay schedule or not (not implemented).")
    parser.add_argument("--max_grad_norm", type=float, default=0, help="Clip the gradients at a certain point.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients across batches.")
    parser.add_argument('--adam_beta1', type=float, default=.9)
    parser.add_argument('--adam_beta2', type=float, default=.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--loss_weighting', nargs="*", default=[])

    ## model params
    #### general model params
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--freeze_embedding_layer', action='store_true')
    parser.add_argument('--freeze_encoder_layers', nargs="*", default=[])
    parser.add_argument('--freeze_pooling_layer', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=.5)
    parser.add_argument('--max_length_seq', type=int, default=512, help='How to generate document embeddings.')
    parser.add_argument('--max_num_sentences', type=int, default=100, help='How to generate document embeddings.')
    parser.add_argument('--max_num_word_positions', type=int, default=1024, help="How many positional embeddings for GPT2.")

    # contextualizing layer
    parser.add_argument('--context_layer', type=str, default='lstm', help="How to provide context to sentence vectors. {lstm, gpt2-sentence}")
    parser.add_argument('--num_contextual_layers', type=int, default=1)
    parser.add_argument('--bidirectional', action='store_true', help="If LSTM only, whether to be bidirectional or not.")
    parser.add_argument('--num_sent_attn_heads', type=int, help="If Transformer context layer only, how many attention heads to have in each layer.")

    # classifier enhancements
    parser.add_argument('--use_positional', action='store_true')
    parser.add_argument('--sinusoidal_embeddings', action='store_true')
    parser.add_argument('--max_position_embeddings', type=int, default=40)
    parser.add_argument('--use_doc_emb', action='store_true')
    parser.add_argument('--doc_embed_arithmetic', action='store_true')

    # head decisions
    parser.add_argument('--do_regression', action='store_true')
    parser.add_argument('--use_poisson_regression', action='store_true')
    return parser