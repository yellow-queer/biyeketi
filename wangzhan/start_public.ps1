# 🌐 公网访问快速启动脚本
# 使用方法：.\start_public.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🍊 柑橘实蝇检测系统 - 公网访问启动" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Ngrok 是否安装
$ngrokPath = Get-Command ngrok -ErrorAction SilentlyContinue

if (-not $ngrokPath) {
    Write-Host "❌ 未检测到 Ngrok，请先安装：" -ForegroundColor Red
    Write-Host "   1. 访问：https://ngrok.com/download" -ForegroundColor Yellow
    Write-Host "   2. 下载 Windows 版本并解压" -ForegroundColor Yellow
    Write-Host "   3. 将 ngrok.exe 添加到 PATH 或放在当前目录" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   或者使用 Chocolatey 安装：" -ForegroundColor Yellow
    Write-Host "   choco install ngrok" -ForegroundColor Cyan
    Write-Host ""
    pause
    exit 1
}

Write-Host "✅ Ngrok 已安装" -ForegroundColor Green
Write-Host ""

# 获取 Ngrok Token
$tokenFile = ".env"
if (Test-Path $tokenFile) {
    $token = Get-Content $tokenFile | Select-String "NGROK_AUTH_TOKEN" | ForEach-Object { $_.Line.Split('=')[1] }
    if ($token) {
        Write-Host "✅ 检测到 Ngrok Token" -ForegroundColor Green
    } else {
        Write-Host "⚠️  未配置 Ngrok Token，请在 .env 文件中添加：" -ForegroundColor Yellow
        Write-Host "   NGROK_AUTH_TOKEN=你的 token" -ForegroundColor Cyan
        Write-Host ""
    }
} else {
    Write-Host "⚠️  未找到 .env 文件" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "📋 启动步骤：" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 步骤 1：启动网站
Write-Host "1️⃣  启动网站服务器..." -ForegroundColor Yellow
Start-Process -FilePath "C:\Users\18656\.conda\envs\chongju\python.exe" -ArgumentList "app.py" -WindowStyle Normal

Start-Sleep -Seconds 3

Write-Host "   ✅ 网站已启动：http://127.0.0.1:5000" -ForegroundColor Green
Write-Host ""

# 步骤 2：启动 Ngrok
Write-Host "2️⃣  启动 Ngrok 内网穿透..." -ForegroundColor Yellow
Write-Host "   （新窗口将打开 Ngrok）" -ForegroundColor Gray
Write-Host ""

Start-Process -FilePath "ngrok" -ArgumentList "http 5000" -WindowStyle Normal

Start-Sleep -Seconds 5

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🎉 启动完成！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📱 访问方式：" -ForegroundColor Cyan
Write-Host "   局域网：http://127.0.0.1:5000" -ForegroundColor White
Write-Host "   手机/平板：http://你的局域网 IP:5000" -ForegroundColor White
Write-Host ""
Write-Host "🌐 公网访问：" -ForegroundColor Cyan
Write-Host "   请在新打开的 Ngrok 窗口中查看公网 URL" -ForegroundColor Yellow
Write-Host "   格式：https://xxxx-xxxx-xxxx.ngrok.io" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 提示：" -ForegroundColor Cyan
Write-Host "   - Ngrok 窗口不要关闭，否则无法公网访问" -ForegroundColor Yellow
Write-Host "   - 每次重启 Ngrok URL 会变化（付费可固定）" -ForegroundColor Yellow
Write-Host "   - 查看详细说明：公网访问配置指南.md" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 显示 Ngrok 状态
Write-Host "🔍 Ngrok 状态检查..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 5 -ErrorAction SilentlyContinue
    $tunnels = $response.Content | ConvertFrom-Json
    
    if ($tunnels.tunnels) {
        foreach ($tunnel in $tunnels.tunnels) {
            if ($tunnel.name -eq "command_line") {
                Write-Host "   ✅ Ngrok 隧道已建立" -ForegroundColor Green
                Write-Host "   公网 URL: $($tunnel.public_url)" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "   📱 现在可以在任何设备上访问：" -ForegroundColor Green
                Write-Host "   $($tunnel.public_url)" -ForegroundColor Cyan
                break
            }
        }
    }
} catch {
    Write-Host "   ⚠️  无法获取 Ngrok 状态，请检查 Ngrok 窗口" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "按任意键退出..." -ForegroundColor Gray
pause > $null
