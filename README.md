# WordPress Meta Broadcast Plugin

A complete WordPress plugin for displaying broadcast notifications with icons, popups, and read tracking functionality.

## 🚀 Features

### 📢 Notification Icon & Badge
- **Customizable icon** with position and color options
- **Animated badge** showing unread notification count
- **Pulse animation** for new notifications
- **Click to open popup** with broadcast list

### 💬 Popup Interface
- **Meta/Instagram-style design** with smooth animations
- **Auto-mark as read** when notifications are viewed
- **Responsive mobile design** with touch-friendly interactions
- **Dark mode support** that adapts to system preferences

### 🔧 Admin Panel
- **Complete broadcast management** interface
- **Rich text editor** for creating broadcasts
- **View and delete** existing broadcasts
- **Usage instructions** with shortcode examples
- **Form validation** and character counting

### 📊 Database Management
- **Two-table structure** for broadcasts and read tracking
- **User-specific read status** with proper foreign key relationships
- **Automatic cleanup** and data integrity

## 📋 Installation

1. Copy the plugin files to your WordPress plugins directory:
   ```
   wp-content/plugins/wp-meta-broadcast/
   ```

2. Activate the plugin through the WordPress admin panel

3. Navigate to **Broadcasts** in the admin menu to create notifications

## 🎯 Usage

### Basic Shortcode
```php
[meta_broadcast_icon]
```
*Default: Fixed position, blue color (#1877f2)*

### With Custom Options
```php
[meta_broadcast_icon position="relative" color="#ff6b6b"]
```

### Available Parameters
- **position**: `fixed` (default) or `relative`
- **color**: Any valid hex color code (default: `#1877f2`)

## 🗂️ File Structure

```
wp-meta-broadcast/
├── wp-meta-broadcast.php     # Main plugin file
├── assets/
│   ├── style.css            # Frontend styles
│   ├── script.js            # Frontend JavaScript
│   ├── admin-style.css      # Admin panel styles
│   └── admin-script.js      # Admin panel JavaScript
└── templates/
    ├── broadcast-list.php   # Popup notification list
    ├── broadcast-form.php   # Form template
    ├── broadcast-icon.php   # Icon and popup structure
    └── admin-page.php       # Admin interface
```

## 📡 AJAX Endpoints

The plugin provides three AJAX endpoints for dynamic functionality:

### `get_unread_count`
Returns the number of unread notifications for the current user.

### `mark_broadcasts_read`
Marks specified notifications as read for the current user.

### `get_broadcasts`
Retrieves all broadcasts with read status, supports popup mode for HTML rendering.

## 🗄️ Database Tables

### `wp_meta_broadcasts`
Stores broadcast notifications:
- `id` - Primary key
- `title` - Notification title
- `content` - Notification content (HTML allowed)
- `created_at` - Creation timestamp
- `updated_at` - Last update timestamp
- `is_active` - Active status (0/1)

### `wp_meta_broadcast_reads`
Tracks read status per user:
- `id` - Primary key
- `user_id` - WordPress user ID
- `broadcast_id` - Foreign key to broadcasts table
- `read_at` - Read timestamp

## 🎨 Customization

### CSS Customization
Override styles by adding custom CSS to your theme:

```css
.meta-broadcast-icon {
    /* Custom icon styles */
}

.meta-broadcast-popup-content {
    /* Custom popup styles */
}
```

### JavaScript Hooks
The plugin provides JavaScript events for custom functionality:

```javascript
// Listen for notification updates
$(document).on('metaBroadcastUpdated', function(e, data) {
    console.log('Notifications updated:', data);
});
```

## 📱 Mobile Support

- **Responsive design** that works on all screen sizes
- **Touch-friendly** interactions with proper touch targets
- **Optimized animations** for mobile performance
- **Reduced motion** support for accessibility

## 🌙 Dark Mode

The plugin automatically detects and adapts to:
- System dark mode preference
- Theme dark mode settings
- Manual dark mode toggles

## 🔒 Security

- **Nonce verification** for all AJAX requests
- **User capability checks** for admin functions
- **SQL injection prevention** with prepared statements
- **XSS protection** with proper data sanitization

## 🚀 Performance

- **Lightweight JavaScript** with minimal dependencies
- **Efficient database queries** with proper indexing
- **Caching-friendly** with cache-busting for assets
- **Auto-refresh intervals** optimized for performance

## 🐛 Troubleshooting

### Common Issues

**Badge not showing:**
- Ensure user is logged in
- Check if there are active broadcasts
- Verify database tables were created during activation

**Popup not opening:**
- Check JavaScript console for errors
- Ensure jQuery is loaded
- Verify AJAX endpoints are accessible

**Admin panel not accessible:**
- Check user has `manage_options` capability
- Verify plugin activation was successful

## 📄 License

This plugin is licensed under the GPL v2 or later.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 🎯 Demo

A live interactive demo is available in the `demo.html` file. The demo showcases:

- **Live notification icon** with badge count
- **Interactive popup** with sample notifications
- **Real-time features** like adding notifications and marking as read
- **Dark mode toggle** to test theme compatibility
- **Mobile responsive** design demonstration

### Demo Features:
- Click the notification icon to see the popup
- Test "Add New Notification" to increase badge count
- Use "Mark All Read" to clear notifications
- Toggle dark mode to see theme adaptation

## 📞 Support

For support and feature requests, please open an issue in the project repository.

---

**Version:** 1.0.0  
**Tested up to:** WordPress 6.4  
**Requires:** WordPress 5.0+, PHP 7.4+