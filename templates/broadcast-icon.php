<?php
/**
 * Template for broadcast icon shortcode
 */

if (!defined('ABSPATH')) {
    exit;
}

$position = esc_attr($atts['position']);
$color = esc_attr($atts['color']);
$user_id = get_current_user_id();
?>

<div class="meta-broadcast-icon-container" data-position="<?php echo $position; ?>" data-color="<?php echo $color; ?>">
    <div class="meta-broadcast-icon" style="background-color: <?php echo $color; ?>;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 10.5C21 10.78 20.78 11 20.5 11H19C18.45 11 18 11.45 18 12V18C18 19.1 17.1 20 16 20H8C6.9 20 6 19.1 6 18V12C6 11.45 5.55 11 5 11H3.5C3.22 11 3 10.78 3 10.5S3.22 10 3.5 10H5C6.1 10 7 10.9 7 12V18C7 18.55 7.45 19 8 19H16C16.55 19 17 18.55 17 18V12C17 10.9 17.9 10 19 10H20.5C20.78 10 21 10.22 21 10.5Z" fill="white"/>
        </svg>
        <span class="meta-broadcast-badge" id="meta-broadcast-badge" style="display: none;">0</span>
    </div>
</div>

<!-- Popup Modal -->
<div id="meta-broadcast-popup" class="meta-broadcast-popup" style="display: none;">
    <div class="meta-broadcast-popup-overlay"></div>
    <div class="meta-broadcast-popup-content">
        <div class="meta-broadcast-popup-header">
            <h3><?php _e('Notifications', 'meta-broadcast'); ?></h3>
            <button class="meta-broadcast-popup-close">&times;</button>
        </div>
        <div class="meta-broadcast-popup-body">
            <div class="meta-broadcast-loading">
                <?php _e('Loading...', 'meta-broadcast'); ?>
            </div>
        </div>
    </div>
</div>