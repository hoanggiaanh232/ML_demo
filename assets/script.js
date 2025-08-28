/**
 * Meta Broadcast Frontend JavaScript
 */

jQuery(document).ready(function($) {
    let unreadBroadcastIds = [];
    
    // Initialize
    updateBadgeCount();
    
    // Auto-refresh badge count every 30 seconds
    setInterval(updateBadgeCount, 30000);
    
    // Icon click handler
    $('.meta-broadcast-icon').on('click', function(e) {
        e.preventDefault();
        openPopup();
    });
    
    // Popup close handlers
    $('.meta-broadcast-popup-close, .meta-broadcast-popup-overlay').on('click', function(e) {
        e.preventDefault();
        closePopup();
    });
    
    // Prevent popup content click from closing
    $('.meta-broadcast-popup-content').on('click', function(e) {
        e.stopPropagation();
    });
    
    // ESC key to close popup
    $(document).on('keydown', function(e) {
        if (e.keyCode === 27 && $('#meta-broadcast-popup').is(':visible')) {
            closePopup();
        }
    });
    
    function updateBadgeCount() {
        if (!metaBroadcast.userId || metaBroadcast.userId === '0') {
            return; // Not logged in
        }
        
        $.ajax({
            url: metaBroadcast.ajaxUrl,
            type: 'POST',
            data: {
                action: 'get_unread_count',
                nonce: metaBroadcast.nonce
            },
            success: function(response) {
                if (response.success) {
                    const count = parseInt(response.data.count);
                    const badge = $('#meta-broadcast-badge');
                    
                    if (count > 0) {
                        badge.text(count > 99 ? '99+' : count).show();
                    } else {
                        badge.hide();
                    }
                }
            },
            error: function() {
                console.warn('Failed to update broadcast badge count');
            }
        });
    }
    
    function openPopup() {
        if (!metaBroadcast.userId || metaBroadcast.userId === '0') {
            alert('Please log in to view notifications.');
            return;
        }
        
        const popup = $('#meta-broadcast-popup');
        const body = $('.meta-broadcast-popup-body');
        
        // Show popup with loading state
        popup.show();
        body.html('<div class="meta-broadcast-loading">Loading...</div>');
        
        // Add body class to prevent scrolling
        $('body').addClass('meta-broadcast-popup-open');
        
        // Load broadcasts
        $.ajax({
            url: metaBroadcast.ajaxUrl,
            type: 'POST',
            data: {
                action: 'get_broadcasts',
                nonce: metaBroadcast.nonce,
                popup_mode: true
            },
            success: function(response) {
                if (response.success) {
                    body.html(response.data.html);
                    
                    // Collect unread broadcast IDs
                    unreadBroadcastIds = [];
                    response.data.broadcasts.forEach(function(broadcast) {
                        if (!broadcast.is_read) {
                            unreadBroadcastIds.push(broadcast.id);
                        }
                    });
                    
                    // Mark broadcasts as read after a short delay
                    if (unreadBroadcastIds.length > 0) {
                        setTimeout(function() {
                            markBroadcastsRead(unreadBroadcastIds);
                        }, 1000);
                    }
                } else {
                    body.html('<div class="meta-broadcast-loading">Error loading notifications.</div>');
                }
            },
            error: function() {
                body.html('<div class="meta-broadcast-loading">Error loading notifications.</div>');
            }
        });
    }
    
    function closePopup() {
        $('#meta-broadcast-popup').fadeOut(200);
        $('body').removeClass('meta-broadcast-popup-open');
    }
    
    function markBroadcastsRead(broadcastIds) {
        if (!broadcastIds.length) return;
        
        $.ajax({
            url: metaBroadcast.ajaxUrl,
            type: 'POST',
            data: {
                action: 'mark_broadcasts_read',
                nonce: metaBroadcast.nonce,
                broadcast_ids: broadcastIds
            },
            success: function(response) {
                if (response.success) {
                    // Update UI to show items as read
                    broadcastIds.forEach(function(id) {
                        $('.meta-broadcast-item[data-id="' + id + '"]')
                            .removeClass('unread')
                            .addClass('read')
                            .find('.meta-broadcast-item-unread-dot')
                            .remove();
                    });
                    
                    // Update badge count
                    updateBadgeCount();
                }
            },
            error: function() {
                console.warn('Failed to mark broadcasts as read');
            }
        });
    }
    
    // Smooth animations
    function animateIcon() {
        const icon = $('.meta-broadcast-icon');
        icon.addClass('animate-bounce');
        setTimeout(function() {
            icon.removeClass('animate-bounce');
        }, 600);
    }
    
    // Add CSS for bounce animation
    $('<style>')
        .text(`
            .meta-broadcast-icon.animate-bounce {
                animation: iconBounce 0.6s ease-in-out;
            }
            
            @keyframes iconBounce {
                0%, 20%, 60%, 100% {
                    transform: translateY(0);
                }
                40% {
                    transform: translateY(-5px);
                }
                80% {
                    transform: translateY(-2px);
                }
            }
        `)
        .appendTo('head');
    
    // Add body class style for popup open state
    $('<style>')
        .text(`
            body.meta-broadcast-popup-open {
                overflow: hidden;
            }
        `)
        .appendTo('head');
    
    // Animate icon when new notifications arrive
    let lastBadgeCount = 0;
    const originalUpdateBadgeCount = updateBadgeCount;
    updateBadgeCount = function() {
        originalUpdateBadgeCount();
        
        setTimeout(function() {
            const currentCount = parseInt($('#meta-broadcast-badge').text()) || 0;
            if (currentCount > lastBadgeCount && lastBadgeCount >= 0) {
                animateIcon();
            }
            lastBadgeCount = currentCount;
        }, 100);
    };
});