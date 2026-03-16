from loss.cross_dice_loss import cross_dice_loss, bce_dice_loss, cross_entropy_loss
from loss.octa_ce_dice_loss import octa_ce_dice_loss


__all__ = {
    'cross_dice_loss': cross_dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'cross_entropy_loss': cross_entropy_loss,
    'octa_ce_dice_loss': octa_ce_dice_loss,
}




def build_loss(hypes):
    name = hypes['loss']['core_method']

    return __all__[name]
