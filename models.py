import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

def load_backbone(name, output_attentions=False):
    if name == 'bert':
        from transformers import BertForMaskedLM, BertTokenizer
        backbone = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=output_attentions)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.name = 'bert-base-uncased'
    elif name == 'roberta':
        from transformers import RobertaForMaskedLM, RobertaTokenizer
        backbone = RobertaForMaskedLM.from_pretrained('roberta-base', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.name = 'roberta-base'
    elif name == 'roberta_large':
        from transformers import RobertaForMaskedLM, RobertaTokenizer
        backbone = RobertaForMaskedLM.from_pretrained('roberta-large', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        tokenizer.name = 'roberta-large'
    elif name == 'roberta_mc':
        from transformers import RobertaForMultipleChoice, RobertaTokenizer
        backbone = RobertaForMultipleChoice.from_pretrained('roberta-base', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.name = 'roberta-base'
    elif name == 'roberta_mc_large':
        from transformers import RobertaForMultipleChoice, RobertaTokenizer
        backbone = RobertaForMultipleChoice.from_pretrained('roberta-large', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        tokenizer.name = 'roberta-large'    
    elif name == 'albert':
        from transformers import AlbertModel, AlbertTokenizer
        backbone = AlbertModel.from_pretrained('albert-base-v2', output_attentions=output_attentions)
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        tokenizer.name = 'albert-base-v2'
    elif name == 'sentence_bert':
        from transformers import AutoTokenizer, AutoModel
        backbone = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        tokenizer.name = 'paraphrase-MiniLM-L6-v2'
    else:
        raise ValueError('No matching backbone network')

    return backbone, tokenizer

class Classifier(nn.Module):
    def __init__(self, backbone_name, backbone, n_classes, train_type='None'):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.dropout = nn.Dropout(0.1)
        self.n_classes = n_classes
        self.train_type = train_type

        if 'large' in backbone_name:
            n_dim = 1024
        elif 'sent' in backbone_name:
            n_dim = 384
        else:
            n_dim = 768
        ##### Classifier for down-stream task
        self.net_cls = nn.Linear(n_dim, n_classes)

    def forward(self, x, inputs_embed=None, get_penul=False, lm=False, sent=False):
        if self.backbone_name in ['bert', 'albert']:
            attention_mask = (x > 0).float()
        else:
            attention_mask = (x != 1).float()

        if inputs_embed is not None:
            out_cls_orig = self.backbone(None, attention_mask=attention_mask, inputs_embeds=inputs_embed)[1]
        elif lm:
            lm_output = self.backbone(x, attention_mask, inputs_embeds=inputs_embed)[0]
            return lm_output
        elif sent:
            out_cls_orig = self.backbone(x, attention_mask, inputs_embeds=inputs_embed)[1]
        else:
            out_cls_orig = self.backbone(x, attention_mask=attention_mask, inputs_embeds=inputs_embed)[1]

        out_cls = self.dropout(out_cls_orig)
        out_cls = self.net_cls(out_cls)
        if self.n_classes == 1:
            out_cls = out_cls.view(-1, 2)
            
        if get_penul:
            return out_cls, out_cls_orig
        else:
            return out_cls

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class Value_estimator(nn.Module):
    # Design choice #1: Deciding what to use as inputs (e.g., BERT final-layer?) -> penultimate of [CLS] token
    # Design choice #2: How to combine the label information? -> naive concatenation

    def __init__(self, args, n_class):
        super(Value_estimator, self).__init__()

        self.n_dim = 768
        self.n_class = n_class

        if args.measurement:
            self.layer1 = nn.Linear(23 + self.n_class, self.n_dim + self.n_class)
        elif args.logit or args.logit2:
            self.layer1 = nn.Linear(self.n_class + self.n_class, self.n_dim + self.n_class)
        else:
            self.layer1 = nn.Linear(self.n_dim + self.n_class, self.n_dim + self.n_class)
        self.layer2 = nn.Linear(self.n_dim + self.n_class, self.n_dim + self.n_class)

        if args.measurement2:
            self.hidden_dim2 = self.n_dim + self.n_class + 23
        else:
            self.hidden_dim2 = self.n_dim + self.n_class + self.n_class

        self.layer3 = nn.Linear(self.hidden_dim2, self.hidden_dim2)
        self.layer4 = nn.Linear(self.hidden_dim2, self.hidden_dim2)
        self.layer5 = nn.Linear(self.hidden_dim2, 1)

    def forward(self, x, y_diff):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x_comb = torch.cat([x, y_diff], dim=-1)
        x_comb = F.relu(self.layer3(x_comb))
        x_comb = F.relu(self.layer4(x_comb))
        x_fin = torch.sigmoid(self.layer5(x_comb))

        return x_fin

class Value_estimator_linear(nn.Module):
    # Design choice #1: Deciding what to use as inputs (e.g., BERT final-layer?) -> penultimate of [CLS] token
    # Design choice #2: How to combine the label information? -> naive concatenation

    def __init__(self, args, n_class):
        super(Value_estimator_linear, self).__init__()

        self.n_dim = 768
        self.n_class = n_class

        if args.measurement:
            self.layer1 = nn.Linear(23 + self.n_class, 1)
        elif args.logit or args.logit2:
            self.layer1 = nn.Linear(self.n_class + self.n_class, 1)
        else:
            self.layer1 = nn.Linear(self.n_dim + self.n_class, 1)

    def forward(self, x, y_diff):
        return torch.sigmoid(self.layer1(x))


##### Codes from the authors () #####
def data_value_evaluator(self):
    """Returns data value evaluator model.
    Here, we assume a simple multi-layer perceptron architecture for the data
    value evaluator model. For data types like tabular, multi-layer perceptron
    is already efficient at extracting the relevant information.

    For high-dimensional data types like images or text,
    it is important to introduce inductive biases to the architecture to
    extract information efficiently. In such cases, there are two options:

    (i) Input the encoded representations (e.g. the last layer activations of
    ResNet for images, or the last layer activations of BERT for  text) and use
    the multi-layer perceptron on top of it. The encoded representations can
    simply come from a pre-trained predictor model using the entire dataset.

    (ii) Modify the data value evaluator model definition below to have the
    appropriate inductive bias (e.g. using convolutional layers for images,
    or attention layers text).
    Returns:
      dve: data value estimations
    """
    with tf.variable_scope('data_value_estimator', reuse=tf.AUTO_REUSE):

      inputs = tf.concat((self.x_input, self.y_input), axis=1)

      # Stacks multi-layered perceptron
      inter_layer = contrib_layers.fully_connected(
          inputs, self.hidden_dim, activation_fn=self.act_fn)
      for _ in range(int(self.layer_number - 3)):
        inter_layer = contrib_layers.fully_connected(
            inter_layer, self.hidden_dim, activation_fn=self.act_fn)
      inter_layer = contrib_layers.fully_connected(
          inter_layer, self.comb_dim, activation_fn=self.act_fn)

      # Combines with y_hat
      # Prediction difference y_hat_input is the prediction difference between predictive models
      # trained on the training set and validation set. (adding y_hat_input into data value estimator as the additional
      # input is observed to improve data value estimation quality in some cases)

      comb_layer = tf.concat((inter_layer, self.y_hat_input), axis=1)
      comb_layer = contrib_layers.fully_connected(
          comb_layer, self.comb_dim, activation_fn=self.act_fn)
      dve = contrib_layers.fully_connected(
          comb_layer, 1, activation_fn=tf.nn.sigmoid)

    return dve

