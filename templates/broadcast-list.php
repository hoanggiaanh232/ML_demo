<?php
/**
 * Template for broadcast list in popup
 */

if (!defined('ABSPATH')) {
    exit;
}
?>

<div class="meta-broadcast-list">
    <?php if (empty($broadcasts)): ?>
        <div class="meta-broadcast-empty">
            <div class="meta-broadcast-empty-icon">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 10.5C21 10.78 20.78 11 20.5 11H19C18.45 11 18 11.45 18 12V18C18 19.1 17.1 20 16 20H8C6.9 20 6 19.1 6 18V12C6 11.45 5.55 11 5 11H3.5C3.22 11 3 10.78 3 10.5S3.22 10 3.5 10H5C6.1 10 7 10.9 7 12V18C7 18.55 7.45 19 8 19H16C16.55 19 17 18.55 17 18V12C17 10.9 17.9 10 19 10H20.5C20.78 10 21 10.22 21 10.5Z" fill="#ccc"/>
                </svg>
            </div>
            <p><?php _e('No notifications yet', 'meta-broadcast'); ?></p>
        </div>
    <?php else: ?>
        <?php foreach ($broadcasts as $broadcast): ?>
            <div class="meta-broadcast-item <?php echo $broadcast->is_read ? 'read' : 'unread'; ?>" data-id="<?php echo $broadcast->id; ?>">
                <div class="meta-broadcast-item-content">
                    <h4 class="meta-broadcast-item-title"><?php echo esc_html($broadcast->title); ?></h4>
                    <div class="meta-broadcast-item-text"><?php echo wp_kses_post($broadcast->content); ?></div>
                    <div class="meta-broadcast-item-meta">
                        <span class="meta-broadcast-item-date">
                            <?php echo human_time_diff(strtotime($broadcast->created_at), current_time('timestamp')) . ' ' . __('ago', 'meta-broadcast'); ?>
                        </span>
                        <?php if (!$broadcast->is_read): ?>
                            <span class="meta-broadcast-item-unread-dot"></span>
                        <?php endif; ?>
                    </div>
                </div>
            </div>
        <?php endforeach; ?>
    <?php endif; ?>
</div>