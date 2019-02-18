## Tensor2tensor manual

### How to install it

```
# Assumes tensorflow or tensorflow-gpu installed
pip install tensor2tensor

# Installs with tensorflow-gpu requirement
pip install tensor2tensor[tensorflow_gpu]

# Installs with tensorflow (cpu) requirement
pip install tensor2tensor[tensorflow]
```

### How to use it

```
# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
t2t-trainer --registry_help

PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR

# Decode

DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> $DECODE_FILE
echo "Goodbye world" >> $DECODE_FILE
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en

# See the translations
cat translation.en

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=translation.en --reference=ref-translation.de
```

### How to add your own components

Components contains : models, hyperparameter sets,modalities and problems.

you can easily add your own components 

```
t2t-trianer
--t2t_usr_dir = your_own_components_dir
```

this is official example
https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/test_data/example_usr_dir

It orgnized like this:

```
__init__.py
my_submodule.py
```

Tensor2tensor will read the __init__.py. In fact __init__.py only contains:

```
from . import my_submodule
```

And the my_submodule.py has

```
"""Example registrations for T2T."""
import re

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry

# Use register_model for a new T2TModel
# Use register_problem for a new Problem
# Use register_hparams for a new hyperparameter set


@registry.register_hparams
def my_very_own_hparams():
  # Start with the base set
  hp = common_hparams.basic_params1()
  # Modify existing hparams
  hp.num_hidden_layers = 2
  # Add new hparams
  hp.add_hparam("filter_size", 2048)
  return hp


@registry.register_problem
class PoetryLines(text_problems.Text2TextProblem):
  """Predict next line of poetry from the last line. From Gutenberg texts."""

  @property
  def approx_vocab_size(self):
    return 2**13  # ~8k

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split

    # pylint: disable=g-import-not-at-top
    from gutenberg import acquire
    from gutenberg import cleanup
    # pylint: enable=g-import-not-at-top

    books = [
        # bookid, skip N lines
        (19221, 223),
        (15553, 522),
    ]

    for (book_id, toskip) in books:
      text = cleanup.strip_headers(acquire.load_etext(book_id)).strip()
      lines = text.split("\n")[toskip:]
      prev_line = None
      ex_count = 0
      for line in lines:
        # Any line that is all upper case is a title or author name
        if not line or line.upper() == line:
          prev_line = None
          continue

        line = re.sub("[^a-z]+", " ", line.strip().lower())
        if prev_line and line:
          yield {
              "inputs": prev_line,
              "targets": line,
          }
          ex_count += 1
        prev_line = line
```

You can find the file register hyperparameters and problem.

### How to define your own problem

Problem means your own datasets.

Google has its own way to specify the problem : [task-family]_[task]_[specifics]

You can generate dataset simply by this code:

```
t2t-datagen \
  --problem=algorithmic_identity_binary40 \
  --data_dir=/tmp
```

And this is the good part of the whole idea

will generate training and development data for the algorithmic copy task - /tmp/algorithmic_identity_binary40-dev-00000-of-00001 and /tmp/algorithmic_identity_binary40-train-00000-of-00001. All tasks produce TFRecord files of tensorflow.Example protocol buffers.

This is saving a lot of your time.

Problem support the generation , trianing and decoding.

1、register your problem. like this

```
@registry.register_problem
class TranslateEndeWmt8k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  @property
  def additional_training_datasets(self):
    """Allow subclasses to add training datasets."""
    return []

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    train_datasets = _ENDE_TRAIN_DATASETS + self.additional_training_datasets
    return train_datasets if train else _ENDE_EVAL_DATASETS

```

2、All the magic happened in problem.generate_data.

Will produce 2 datasets. Problem.training_filepaths for train and Problem.dev_filepaths for dev. And other files eg.vocabulary file.

one easy way is implement way is create training data and dev data of 2 generators.Then pass to generator_utils.generate_dataset_and_shuffle.

```
def generate_data(self, data_dir, tmp_dir, task_id=-1):

    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    split_paths = [(split["split"], filepath_fns[split["split"]](
        data_dir, split["shards"], shuffled=self.already_shuffled))
                   for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    if self.is_generate_per_split:
      for split, paths in split_paths:
        generator_utils.generate_files(
            self.generate_encoded_samples(data_dir, tmp_dir, split), paths)
    else:
      generator_utils.generate_files(
          self.generate_encoded_samples(
              data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

    generator_utils.shuffle_dataset(all_paths, extra_fn=self._pack_fn())

```

Because the translate is so commam. Tensor2tensor team has change it a lot. But you still can see the shuffle_dataset.In fact, about the translate problem.You only specify the vocab_size and data size. So I must find another example.

```
 def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    test_paths = self.test_filepaths(
        data_dir, self.num_test_shards, shuffled=True)

    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir, self.TEST_DATASETS), test_paths)

    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), all_paths)
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), train_paths,
          self.generator(data_dir, tmp_dir, self.DEV_DATASETS), dev_paths)
```

You can find there is two generator passed to the generate_dataset_and_shuffle.

3、How generator works

Generators yield dics with string keys and values being lists of {int,float,str}

this is how it works:

```
def length_generator(nbr_cases):
  for _ in range(nbr_cases):
    length = np.random.randint(100) + 1
    yield {"inputs": [2] * length, "targets": [length]}
```

note: Do not use 0 and 1. 0 for pad. 1 for the end of sentense.

TODO: Downlaod,unzip data. 

### How to define modality

### How to define your own model

Model take dense tensors in and produce dense tensors which may use modality

All pre-made models in models subpackage.

Models inherit from T2TModel

Models register with

@registry.register_model

### How to define your own Hyperparameter sets

### Others

### source

The code base
https://github.com/tensorflow/tensor2tensor
