# Installation Guide - WordPress Meta Broadcast Plugin

## Quick Setup

### 1. Upload Plugin Files
Copy all plugin files to your WordPress installation:
```
wp-content/plugins/wp-meta-broadcast/
```

### 2. Activate Plugin
1. Go to WordPress Admin → Plugins
2. Find "Meta Broadcast" in the plugin list
3. Click "Activate"

### 3. Create Your First Broadcast
1. Navigate to **Broadcasts** in the admin menu
2. Fill in the title and content
3. Click "Create Broadcast"

### 4. Add Icon to Your Site
Add the shortcode to any post, page, or widget:
```
[meta_broadcast_icon]
```

## Customization Options

### Icon Position & Color
```php
// Fixed position (default)
[meta_broadcast_icon]

// Relative position with custom color
[meta_broadcast_icon position="relative" color="#ff6b6b"]

// Custom color only
[meta_broadcast_icon color="#28a745"]
```

### CSS Customization
Add to your theme's CSS:
```css
/* Customize icon appearance */
.meta-broadcast-icon {
    width: 60px;
    height: 60px;
    /* Add your custom styles */
}

/* Customize popup appearance */
.meta-broadcast-popup-content {
    border-radius: 20px;
    /* Add your custom styles */
}
```

## Database Tables

The plugin automatically creates these tables:

### wp_meta_broadcasts
- Stores notification content
- Fields: id, title, content, created_at, updated_at, is_active

### wp_meta_broadcast_reads  
- Tracks user read status
- Fields: id, user_id, broadcast_id, read_at

## Troubleshooting

### Common Issues

**No notifications showing:**
- Check if broadcasts are marked as active
- Verify user is logged in
- Check browser console for JavaScript errors

**Badge not updating:**
- Clear browser cache
- Check AJAX endpoints are accessible
- Verify nonce verification is working

**Admin panel not accessible:**
- Ensure user has `manage_options` capability
- Check plugin activation was successful
- Verify database tables were created

### Debug Mode
Add to wp-config.php for debugging:
```php
define('WP_DEBUG', true);
define('WP_DEBUG_LOG', true);
```

## Performance Optimization

### Caching
- Plugin is cache-friendly
- AJAX calls bypass cache automatically
- Static assets include version strings

### Database Optimization
- Tables use proper indexes
- Old read records can be cleaned periodically
- Queries are optimized for performance

## Security Features

- ✅ Nonce verification for all AJAX requests
- ✅ User capability checks for admin functions  
- ✅ SQL injection prevention with prepared statements
- ✅ XSS protection with data sanitization
- ✅ CSRF protection on forms

## Uninstallation

To completely remove the plugin:

1. Deactivate the plugin
2. Delete plugin files
3. Optionally remove database tables:
```sql
DROP TABLE wp_meta_broadcasts;
DROP TABLE wp_meta_broadcast_reads;
```

---

For more help, see the main README.md file or open a support issue.