import stanza
stanza.download("en")
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

model = 'stanza'

if model == 'stanza' or model == 'spacy':

    configuration = {
        "nlp_engine_name": "stanza",
        "models": [{"lang_code": "en", "model_name": "en"}]
    }
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_md"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
else:
    conf_file = './presidio_classic.json'

    # Create NLP engine based on configuration
    provider = NlpEngineProvider(conf_file=conf_file)
    nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine,
                          supported_languages=["en"])
anonymizer = AnonymizerEngine()