from rest_framework.routers import SimpleRouter

from .views.artwork import ArtworkViewSet
from .views.artwork_purchase import ArtworkPurchaseViewSet
from .views.artwork_transfer import ArtworkTransferViewSet
from .views.openai_image import OpenAIImageViewSet

router = SimpleRouter(trailing_slash=False)
router.register('artworks', ArtworkViewSet)
router.register('artwork_purchases', ArtworkPurchaseViewSet)
router.register('artwork_transfers', ArtworkTransferViewSet)
router.register('openai_images', OpenAIImageViewSet, basename='openai-images')

urlpatterns = router.urls
