Param(
    [string]$UserPrompt = "Hello! Can you summarize how this local endpoint works?"
)

$uri = "http://127.0.0.1:54546/v1/chat/completions"

$payload = @{
    model    = "gemma-3-270m-it"
    messages = @(
        @{
            role    = "system"
            content = "You are a concise, helpful assistant."
        },
        @{
            role    = "user"
            content = $UserPrompt
        }
    )
    max_tokens  = 16000
    temperature = 0.7
} | ConvertTo-Json -Depth 4

try {
    $response = Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body $payload
    $reply = $response.choices[0].message.content.Trim()
    Write-Host "Assistant:" -ForegroundColor Cyan
    Write-Host $reply
}
catch {
    Write-Error "Request failed: $($_.Exception.Message)"
    if ($_.Exception.Response -and $_.Exception.Response.ContentLength -gt 0) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        Write-Error "Server response:`n$($reader.ReadToEnd())"
    }
    exit 1
}
