/**
 * Meta Broadcast Admin JavaScript
 */

jQuery(document).ready(function($) {
    // Admin functionality can be added here
    
    // Form validation
    $('#meta-broadcast-form').on('submit', function(e) {
        const title = $('#broadcast-title').val().trim();
        const content = $('#broadcast-content').val().trim();
        
        if (!title || !content) {
            e.preventDefault();
            alert('Please fill in all required fields.');
            return false;
        }
    });
    
    // Auto-resize textareas
    $('textarea').on('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Confirm delete actions
    $('.delete-broadcast').on('click', function(e) {
        if (!confirm('Are you sure you want to delete this broadcast?')) {
            e.preventDefault();
            return false;
        }
    });
    
    // Character counter for title (optional enhancement)
    $('#broadcast_title').on('input', function() {
        const maxLength = 255;
        const currentLength = $(this).val().length;
        const remaining = maxLength - currentLength;
        
        let counter = $(this).next('.character-counter');
        if (counter.length === 0) {
            counter = $('<small class="character-counter"></small>');
            $(this).after(counter);
        }
        
        counter.text(remaining + ' characters remaining');
        
        if (remaining < 20) {
            counter.css('color', '#d63638');
        } else if (remaining < 50) {
            counter.css('color', '#dba617');
        } else {
            counter.css('color', '#666');
        }
    });
    
    // Add some admin-specific styles
    $('<style>')
        .text(`
            .character-counter {
                display: block;
                margin-top: 5px;
                font-style: italic;
            }
            
            .meta-broadcast-admin-container .form-table th {
                width: 200px;
            }
            
            .meta-broadcast-admin-container .wp-list-table th,
            .meta-broadcast-admin-container .wp-list-table td {
                vertical-align: top;
            }
            
            .meta-broadcast-admin-container .wp-list-table td:nth-child(2) {
                max-width: 300px;
                word-wrap: break-word;
            }
        `)
        .appendTo('head');
});