def update_response_headers(response):
    ContentSecurityPolicy = ''
    ContentSecurityPolicy += "default-src 'self'; "
    ContentSecurityPolicy += "script-src 'self' cdnjs.cloudflare.com 'unsafe-inline' 'unsafe-eval' ; "
    ContentSecurityPolicy += "style-src 'self' cdnjs.cloudflare.com 'unsafe-inline' ; "
    ContentSecurityPolicy += "font-src 'self' cdnjs.cloudflare.com 'unsafe-inline' ; "
    #ContentSecurityPolicy += "script-src 'self' 'unsafe-inline'; "
    #ContentSecurityPolicy += "style-src 'self' 'unsafe-inline'; "
    #ContentSecurityPolicy += "img-src 'self' data:; "
    #ContentSecurityPolicy += "connect-src 'self';"
    response.headers.add('Content-Security-Policy', ContentSecurityPolicy)
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('Strict-Transport-Security', 'max-age=86400; includeSubDomains')
    response.headers.add('X-Frame-Options', 'deny')
    #response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    response.headers.set('Server', '')
    response.headers.set('Referrer-policy','same-origin')


    #response.headers.add('Access-Control-Allow-Origin', '*')

    return response