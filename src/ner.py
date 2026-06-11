from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import spacy
from collections import Counter


class Ner:
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")

    def ne_chunk_entities(self):
        tokens = word_tokenize(self.text)
        pos_tags = pos_tag(tokens)
        entities = ne_chunk(pos_tags)

        entity_list = []
        for entity in entities:
            if hasattr(entity, "label"):
                entity_name = " ".join([word for word, tag in entity])

                entity_type = entity.label()

                entity_list.append((entity_name, entity_type))

        print(entity_list)
        return entity_list

    def spacy_ner(self):
        self.doc = self.nlp(self.text)
        self.entity_list_2 = []

        for ent in self.doc.ents:
            self.entity_list_2.append((ent.text, ent.label_))

        print(self.entity_list_2[0:6])
        return self.entity_list_2

    def spacy_entities_freq(self):
        entities = self.spacy_ner()
        name_counter = Counter([name for name, label in entities])
        print(name_counter.most_common(6))

