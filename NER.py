from gliner import GLiNER


class NamedEntityRecognizer:
    def __init__(self):
        self.model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
        print("GLiNER model loaded.")

    def predict_entities(self, text, labels):
        entities = self.model.predict_entities(text, labels,threshold=0.15 )
        return entities

