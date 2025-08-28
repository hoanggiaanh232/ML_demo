<?php
/**
 * Plugin Name: Meta Broadcast
 * Plugin URI: https://example.com/wp-meta-broadcast
 * Description: A WordPress plugin to display broadcast notifications with icons, popups, and read tracking.
 * Version: 1.0.0
 * Author: Your Name
 * License: GPL v2 or later
 * Text Domain: meta-broadcast
 */

// Prevent direct access
if (!defined('ABSPATH')) {
    exit;
}

// Define plugin constants
define('META_BROADCAST_VERSION', '1.0.0');
define('META_BROADCAST_PLUGIN_URL', plugin_dir_url(__FILE__));
define('META_BROADCAST_PLUGIN_PATH', plugin_dir_path(__FILE__));

class MetaBroadcast {
    
    public function __construct() {
        register_activation_hook(__FILE__, array($this, 'activate'));
        register_deactivation_hook(__FILE__, array($this, 'deactivate'));
        
        add_action('init', array($this, 'init'));
        add_action('wp_enqueue_scripts', array($this, 'enqueue_scripts'));
        add_action('admin_enqueue_scripts', array($this, 'admin_enqueue_scripts'));
        add_action('admin_menu', array($this, 'admin_menu'));
        
        // AJAX handlers
        add_action('wp_ajax_get_unread_count', array($this, 'get_unread_count'));
        add_action('wp_ajax_nopriv_get_unread_count', array($this, 'get_unread_count'));
        add_action('wp_ajax_mark_broadcasts_read', array($this, 'mark_broadcasts_read'));
        add_action('wp_ajax_nopriv_mark_broadcasts_read', array($this, 'mark_broadcasts_read'));
        add_action('wp_ajax_get_broadcasts', array($this, 'get_broadcasts'));
        add_action('wp_ajax_nopriv_get_broadcasts', array($this, 'get_broadcasts'));
        
        // Shortcode
        add_shortcode('meta_broadcast_icon', array($this, 'broadcast_icon_shortcode'));
    }
    
    public function activate() {
        $this->create_tables();
        flush_rewrite_rules();
    }
    
    public function deactivate() {
        flush_rewrite_rules();
    }
    
    public function init() {
        load_plugin_textdomain('meta-broadcast', false, dirname(plugin_basename(__FILE__)) . '/languages');
    }
    
    public function create_tables() {
        global $wpdb;
        
        $charset_collate = $wpdb->get_charset_collate();
        
        // Create broadcasts table
        $broadcasts_table = $wpdb->prefix . 'meta_broadcasts';
        $sql_broadcasts = "CREATE TABLE $broadcasts_table (
            id int(11) NOT NULL AUTO_INCREMENT,
            title varchar(255) NOT NULL,
            content text NOT NULL,
            created_at datetime DEFAULT CURRENT_TIMESTAMP,
            updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_active tinyint(1) DEFAULT 1,
            PRIMARY KEY (id)
        ) $charset_collate;";
        
        // Create broadcast reads table
        $reads_table = $wpdb->prefix . 'meta_broadcast_reads';
        $sql_reads = "CREATE TABLE $reads_table (
            id int(11) NOT NULL AUTO_INCREMENT,
            user_id int(11) NOT NULL,
            broadcast_id int(11) NOT NULL,
            read_at datetime DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id),
            UNIQUE KEY user_broadcast (user_id, broadcast_id),
            FOREIGN KEY (broadcast_id) REFERENCES $broadcasts_table(id) ON DELETE CASCADE
        ) $charset_collate;";
        
        require_once(ABSPATH . 'wp-admin/includes/upgrade.php');
        dbDelta($sql_broadcasts);
        dbDelta($sql_reads);
    }
    
    public function enqueue_scripts() {
        wp_enqueue_style('meta-broadcast-style', META_BROADCAST_PLUGIN_URL . 'assets/style.css', array(), META_BROADCAST_VERSION);
        wp_enqueue_script('meta-broadcast-script', META_BROADCAST_PLUGIN_URL . 'assets/script.js', array('jquery'), META_BROADCAST_VERSION, true);
        
        wp_localize_script('meta-broadcast-script', 'metaBroadcast', array(
            'ajaxUrl' => admin_url('admin-ajax.php'),
            'nonce' => wp_create_nonce('meta_broadcast_nonce'),
            'userId' => get_current_user_id()
        ));
    }
    
    public function admin_enqueue_scripts($hook) {
        if (strpos($hook, 'meta-broadcast') !== false) {
            wp_enqueue_style('meta-broadcast-admin-style', META_BROADCAST_PLUGIN_URL . 'assets/admin-style.css', array(), META_BROADCAST_VERSION);
            wp_enqueue_script('meta-broadcast-admin-script', META_BROADCAST_PLUGIN_URL . 'assets/admin-script.js', array('jquery'), META_BROADCAST_VERSION, true);
        }
    }
    
    public function admin_menu() {
        add_menu_page(
            __('Meta Broadcasts', 'meta-broadcast'),
            __('Broadcasts', 'meta-broadcast'),
            'manage_options',
            'meta-broadcast',
            array($this, 'admin_page'),
            'dashicons-megaphone',
            30
        );
    }
    
    public function admin_page() {
        include META_BROADCAST_PLUGIN_PATH . 'templates/admin-page.php';
    }
    
    public function broadcast_icon_shortcode($atts) {
        $atts = shortcode_atts(array(
            'position' => 'fixed',
            'color' => '#1877f2'
        ), $atts);
        
        ob_start();
        include META_BROADCAST_PLUGIN_PATH . 'templates/broadcast-icon.php';
        return ob_get_clean();
    }
    
    public function get_unread_count() {
        if (!wp_verify_nonce($_POST['nonce'], 'meta_broadcast_nonce')) {
            wp_die('Security check failed');
        }
        
        global $wpdb;
        $user_id = get_current_user_id();
        
        if (!$user_id) {
            wp_send_json_error('User not logged in');
            return;
        }
        
        $broadcasts_table = $wpdb->prefix . 'meta_broadcasts';
        $reads_table = $wpdb->prefix . 'meta_broadcast_reads';
        
        $unread_count = $wpdb->get_var($wpdb->prepare("
            SELECT COUNT(*) 
            FROM $broadcasts_table b 
            LEFT JOIN $reads_table r ON b.id = r.broadcast_id AND r.user_id = %d 
            WHERE b.is_active = 1 AND r.id IS NULL
        ", $user_id));
        
        wp_send_json_success(array('count' => intval($unread_count)));
    }
    
    public function mark_broadcasts_read() {
        if (!wp_verify_nonce($_POST['nonce'], 'meta_broadcast_nonce')) {
            wp_die('Security check failed');
        }
        
        global $wpdb;
        $user_id = get_current_user_id();
        
        if (!$user_id) {
            wp_send_json_error('User not logged in');
            return;
        }
        
        $broadcast_ids = isset($_POST['broadcast_ids']) ? array_map('intval', $_POST['broadcast_ids']) : array();
        
        if (empty($broadcast_ids)) {
            wp_send_json_error('No broadcast IDs provided');
            return;
        }
        
        $reads_table = $wpdb->prefix . 'meta_broadcast_reads';
        
        foreach ($broadcast_ids as $broadcast_id) {
            $wpdb->replace($reads_table, array(
                'user_id' => $user_id,
                'broadcast_id' => $broadcast_id,
                'read_at' => current_time('mysql')
            ));
        }
        
        wp_send_json_success('Broadcasts marked as read');
    }
    
    public function get_broadcasts() {
        if (!wp_verify_nonce($_POST['nonce'], 'meta_broadcast_nonce')) {
            wp_die('Security check failed');
        }
        
        global $wpdb;
        $user_id = get_current_user_id();
        $popup_mode = isset($_POST['popup_mode']) ? $_POST['popup_mode'] : false;
        
        $broadcasts_table = $wpdb->prefix . 'meta_broadcasts';
        $reads_table = $wpdb->prefix . 'meta_broadcast_reads';
        
        $sql = "
            SELECT b.*, 
                   CASE WHEN r.id IS NOT NULL THEN 1 ELSE 0 END as is_read,
                   r.read_at
            FROM $broadcasts_table b 
            LEFT JOIN $reads_table r ON b.id = r.broadcast_id AND r.user_id = %d 
            WHERE b.is_active = 1 
            ORDER BY b.created_at DESC
        ";
        
        $broadcasts = $wpdb->get_results($wpdb->prepare($sql, $user_id));
        
        if ($popup_mode) {
            ob_start();
            include META_BROADCAST_PLUGIN_PATH . 'templates/broadcast-list.php';
            $html = ob_get_clean();
            
            wp_send_json_success(array(
                'html' => $html,
                'broadcasts' => $broadcasts
            ));
        } else {
            wp_send_json_success(array('broadcasts' => $broadcasts));
        }
    }
}

// Initialize the plugin
new MetaBroadcast();