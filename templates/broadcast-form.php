<?php
/**
 * Template for broadcast form (for potential future use)
 */

if (!defined('ABSPATH')) {
    exit;
}
?>

<div class="meta-broadcast-form">
    <form id="meta-broadcast-form" method="post">
        <div class="form-group">
            <label for="broadcast-title"><?php _e('Title', 'meta-broadcast'); ?></label>
            <input type="text" id="broadcast-title" name="title" required />
        </div>
        
        <div class="form-group">
            <label for="broadcast-content"><?php _e('Content', 'meta-broadcast'); ?></label>
            <textarea id="broadcast-content" name="content" rows="5" required></textarea>
        </div>
        
        <div class="form-actions">
            <button type="submit" class="button button-primary">
                <?php _e('Save Broadcast', 'meta-broadcast'); ?>
            </button>
        </div>
    </form>
</div>