from django.core.management.base import BaseCommand
from delphic.utils.collections import get_embedding_model, get_inference_model


class Command(BaseCommand):
    help = 'download embedding and inference models'

    def handle(self, *args, **kwargs):
        get_embedding_model()
        print("successfully downloaded embedding models")

        get_inference_model()
        print("successfully downloaded inference models")
