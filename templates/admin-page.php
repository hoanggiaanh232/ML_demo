<?php
/**
 * Admin page template
 */

if (!defined('ABSPATH')) {
    exit;
}

// Handle form submissions
if (isset($_POST['submit_broadcast'])) {
    if (wp_verify_nonce($_POST['broadcast_nonce'], 'create_broadcast')) {
        global $wpdb;
        
        $title = sanitize_text_field($_POST['broadcast_title']);
        $content = wp_kses_post($_POST['broadcast_content']);
        
        if (!empty($title) && !empty($content)) {
            $result = $wpdb->insert(
                $wpdb->prefix . 'meta_broadcasts',
                array(
                    'title' => $title,
                    'content' => $content,
                    'created_at' => current_time('mysql'),
                    'is_active' => 1
                )
            );
            
            if ($result) {
                echo '<div class="notice notice-success"><p>' . __('Broadcast created successfully!', 'meta-broadcast') . '</p></div>';
            } else {
                echo '<div class="notice notice-error"><p>' . __('Error creating broadcast.', 'meta-broadcast') . '</p></div>';
            }
        } else {
            echo '<div class="notice notice-error"><p>' . __('Please fill in all fields.', 'meta-broadcast') . '</p></div>';
        }
    }
}

// Handle broadcast deletion
if (isset($_GET['action']) && $_GET['action'] === 'delete' && isset($_GET['id'])) {
    if (wp_verify_nonce($_GET['_wpnonce'], 'delete_broadcast_' . $_GET['id'])) {
        global $wpdb;
        $wpdb->update(
            $wpdb->prefix . 'meta_broadcasts',
            array('is_active' => 0),
            array('id' => intval($_GET['id']))
        );
        echo '<div class="notice notice-success"><p>' . __('Broadcast deleted successfully!', 'meta-broadcast') . '</p></div>';
    }
}

// Get existing broadcasts
global $wpdb;
$broadcasts = $wpdb->get_results("SELECT * FROM {$wpdb->prefix}meta_broadcasts WHERE is_active = 1 ORDER BY created_at DESC");
?>

<div class="wrap">
    <h1><?php _e('Meta Broadcasts', 'meta-broadcast'); ?></h1>
    
    <div class="meta-broadcast-admin-container">
        <!-- Create New Broadcast Form -->
        <div class="meta-broadcast-form-section">
            <h2><?php _e('Create New Broadcast', 'meta-broadcast'); ?></h2>
            <form method="post" action="">
                <?php wp_nonce_field('create_broadcast', 'broadcast_nonce'); ?>
                
                <table class="form-table">
                    <tr>
                        <th scope="row">
                            <label for="broadcast_title"><?php _e('Title', 'meta-broadcast'); ?></label>
                        </th>
                        <td>
                            <input type="text" id="broadcast_title" name="broadcast_title" class="regular-text" required />
                        </td>
                    </tr>
                    <tr>
                        <th scope="row">
                            <label for="broadcast_content"><?php _e('Content', 'meta-broadcast'); ?></label>
                        </th>
                        <td>
                            <?php 
                            wp_editor('', 'broadcast_content', array(
                                'textarea_name' => 'broadcast_content',
                                'media_buttons' => false,
                                'textarea_rows' => 10,
                                'teeny' => true
                            )); 
                            ?>
                        </td>
                    </tr>
                </table>
                
                <?php submit_button(__('Create Broadcast', 'meta-broadcast'), 'primary', 'submit_broadcast'); ?>
            </form>
        </div>
        
        <!-- Existing Broadcasts -->
        <div class="meta-broadcast-list-section">
            <h2><?php _e('Existing Broadcasts', 'meta-broadcast'); ?></h2>
            
            <?php if (empty($broadcasts)): ?>
                <p><?php _e('No broadcasts found.', 'meta-broadcast'); ?></p>
            <?php else: ?>
                <table class="wp-list-table widefat fixed striped">
                    <thead>
                        <tr>
                            <th scope="col"><?php _e('Title', 'meta-broadcast'); ?></th>
                            <th scope="col"><?php _e('Content', 'meta-broadcast'); ?></th>
                            <th scope="col"><?php _e('Created', 'meta-broadcast'); ?></th>
                            <th scope="col"><?php _e('Actions', 'meta-broadcast'); ?></th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($broadcasts as $broadcast): ?>
                            <tr>
                                <td><strong><?php echo esc_html($broadcast->title); ?></strong></td>
                                <td><?php echo wp_trim_words(wp_strip_all_tags($broadcast->content), 15); ?></td>
                                <td><?php echo date('Y-m-d H:i', strtotime($broadcast->created_at)); ?></td>
                                <td>
                                    <a href="<?php echo wp_nonce_url(add_query_arg(array('action' => 'delete', 'id' => $broadcast->id)), 'delete_broadcast_' . $broadcast->id); ?>" 
                                       onclick="return confirm('<?php _e('Are you sure you want to delete this broadcast?', 'meta-broadcast'); ?>')"
                                       class="button button-secondary">
                                        <?php _e('Delete', 'meta-broadcast'); ?>
                                    </a>
                                </td>
                            </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            <?php endif; ?>
        </div>
        
        <!-- Usage Instructions -->
        <div class="meta-broadcast-usage-section">
            <h2><?php _e('Usage Instructions', 'meta-broadcast'); ?></h2>
            <div class="card">
                <h3><?php _e('Shortcode Usage', 'meta-broadcast'); ?></h3>
                <p><?php _e('Use the following shortcode to display the broadcast icon:', 'meta-broadcast'); ?></p>
                <code>[meta_broadcast_icon]</code>
                
                <h4><?php _e('Available Parameters:', 'meta-broadcast'); ?></h4>
                <ul>
                    <li><strong>position</strong>: fixed (default) or relative</li>
                    <li><strong>color</strong>: hex color code (default: #1877f2)</li>
                </ul>
                
                <h4><?php _e('Examples:', 'meta-broadcast'); ?></h4>
                <p><code>[meta_broadcast_icon position="relative" color="#ff6b6b"]</code></p>
                <p><code>[meta_broadcast_icon color="#28a745"]</code></p>
            </div>
        </div>
    </div>
</div>