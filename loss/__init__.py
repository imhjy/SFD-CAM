from loss.octa_ce_dice_loss import octa_ce_dice_loss


__all__ = {
    'octa_ce_dice_loss': octa_ce_dice_loss,
}




def build_loss(hypes):
    name = hypes['loss']['core_method']

    return __all__[name]
