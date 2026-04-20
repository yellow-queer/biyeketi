/**
 * 柑橘检测多智能体网站 - 前端交互逻辑
 */

// ==================== 全局变量 ====================
// API 基础 URL（自动获取当前域名，支持本地和 Ngrok 访问）
const API_BASE_URL = window.location.protocol + '//' + window.location.host;
console.log('🔧 API 基础 URL:', API_BASE_URL);

let currentImageFile = null;
let currentImageBase64 = null;
let conversationId = 'default_' + Date.now();
let isSearching = false;
let isRAGMode = false;

// ==================== DOM 元素 ====================
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const btnRemove = document.getElementById('btnRemove');
const resultContainer = document.getElementById('resultContainer');
const btnDetect = document.getElementById('btnDetect');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const btnSend = document.getElementById('btnSend');
const btnSearch = document.getElementById('btnSearch');
const btnRAG = document.getElementById('btnRAG');
const btnClear = document.getElementById('btnClear');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');

// 质量评估相关元素
const qualityAssessment = document.getElementById('qualityAssessment');
const qualityScoreEl = document.getElementById('qualityScore');
const sharpnessValueEl = document.getElementById('sharpnessValue');
const brightnessValueEl = document.getElementById('brightnessValue');
const contrastValueEl = document.getElementById('contrastValue');
const colorValueEl = document.getElementById('colorValue');
const qualitySuggestions = document.getElementById('qualitySuggestions');
const suggestionsList = document.getElementById('suggestionsList');

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    initUploadArea();
    initChatInput();
    initButtons();
    checkSystemStatus();
});

// ==================== 系统状态检查 ====================
async function checkSystemStatus() {
    try {
        const response = await fetch(API_BASE_URL + '/api/status');
        const data = await response.json();
        if (data.success) {
            console.log('✓ 系统状态正常');
        }
    } catch (error) {
        console.error('✗ 系统状态检查失败:', error);
    }
}

// ==================== 上传区域初始化 ====================
function initUploadArea() {
    // 点击上传
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });

    // 文件选择
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelect(file);
        } else {
            showMessage('请上传图像文件', 'error');
        }
    });

    // 移除图像
    btnRemove.addEventListener('click', () => {
        resetUpload();
    });
}

// ==================== 处理文件选择 ====================
function handleFileSelect(file) {
    // 验证文件类型
    if (!file.type.startsWith('image/')) {
        showMessage('请上传有效的图像文件', 'error');
        return;
    }

    // 验证文件大小（最大 10MB）
    if (file.size > 10 * 1024 * 1024) {
        showMessage('图像大小不能超过 10MB', 'error');
        return;
    }

    currentImageFile = file;

    // 读取并预览图像
    const reader = new FileReader();
    reader.onload = async (e) => {
        currentImageBase64 = e.target.result.split(',')[1]; // 移除 data:image/...;base64,前缀
        imagePreview.src = e.target.result;
        
        // 显示预览，隐藏上传区域
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        resultContainer.style.display = 'none';
        
        // 启用检测按钮
        btnDetect.disabled = false;
        
        // 自动进行质量评估
        await assessImageQuality(file);
    };
    reader.readAsDataURL(file);
}

// ==================== 重置上传 ====================
function resetUpload() {
    currentImageFile = null;
    currentImageBase64 = null;
    imageInput.value = '';
    uploadArea.style.display = 'block';
    previewContainer.style.display = 'none';
    resultContainer.style.display = 'none';
    btnDetect.disabled = true;
}

// ==================== 检测按钮初始化 ====================
function initButtons() {
    // 检测按钮
    btnDetect.addEventListener('click', async () => {
        if (!currentImageBase64) {
            showMessage('请先上传图像', 'error');
            return;
        }
        await performDetection();
    });

    // 发送按钮
    btnSend.addEventListener('click', sendMessage);

    // 回车发送
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 搜索按钮
    btnSearch.addEventListener('click', toggleSearchMode);

    // RAG 按钮
    btnRAG.addEventListener('click', toggleRAGMode);

    // 清除按钮
    btnClear.addEventListener('click', clearConversation);
}

// ==================== 执行检测 ====================
async function performDetection() {
    showLoading('正在检测中...');
    
    try {
        if (!currentImageFile) {
            showMessage('请先选择图像', 'error');
            hideLoading();
            return;
        }
        
        const formData = new FormData();
        formData.append('image', currentImageFile);

        // 使用完整的 API URL（支持本地和 Ngrok 访问）
        const apiUrl = API_BASE_URL + '/api/detection/predict';
        console.log('📤 发送检测请求到:', apiUrl);
        console.log('📁 文件大小:', (currentImageFile.size / 1024 / 1024).toFixed(2), 'MB');
        
        // 设置超时时间
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 秒超时
        
        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);

        // 检查响应状态
        if (!response.ok) {
            throw new Error(`HTTP 错误：${response.status} ${response.statusText}`);
        }

        // 检查响应类型
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            console.error('❌ 非 JSON 响应，响应内容:', text.substring(0, 500));
            throw new Error('服务器返回了非 JSON 响应，可能是网络错误或 API 路径错误。请检查：1. Ngrok 是否正常运行 2. 服务器是否正常运行 3. 刷新页面后重试');
        }

        const data = await response.json();
        console.log('✅ 检测结果:', data);
        
        if (data.success) {
            displayDetectionResult(data.result);
            
            // 自动发送到对话系统
            const detectionMessage = `我上传了一张柑橘图像，检测结果显示为${data.result.class}，置信度${(data.result.confidence * 100).toFixed(2)}%。`;
            await sendChatMessage(detectionMessage, data.result);
        } else {
            showMessage(data.message || '检测失败', 'error');
        }
    } catch (error) {
        console.error('❌ 检测错误:', error);
        let errorMsg = '检测失败，请重试';
        
        // 提供更详细的错误信息
        if (error.name === 'AbortError') {
            errorMsg = '请求超时，请重试（图像可能过大）';
        } else if (error.message.includes('非 JSON')) {
            errorMsg = error.message;
        } else if (error.message.includes('Failed to fetch')) {
            errorMsg = '无法连接到服务器，请检查：\n1. 网络连接是否正常\n2. Ngrok 是否正常运行\n3. 服务器是否已启动';
        } else if (error.message.includes('HTTP 错误')) {
            errorMsg = `服务器响应错误：${error.message}，请刷新页面后重试`;
        }
        
        showMessage(errorMsg, 'error');
    } finally {
        hideLoading();
    }
}

// ==================== 显示检测结果 ====================
function displayDetectionResult(result) {
    // 更新徽章
    const badge = document.getElementById('resultBadge');
    badge.textContent = result.class;
    badge.className = 'result-badge ' + (result.class === '健康' ? 'healthy' : 'diseased');

    // 更新进度条
    const healthyPercent = (result.probabilities.健康 * 100).toFixed(1);
    const diseasedPercent = (result.probabilities.患虫 * 100).toFixed(1);

    document.getElementById('healthyBar').style.width = healthyPercent + '%';
    document.getElementById('healthyValue').textContent = healthyPercent + '%';
    document.getElementById('diseasedBar').style.width = diseasedPercent + '%';
    document.getElementById('diseasedValue').textContent = diseasedPercent + '%';

    // 更新总结
    const summary = document.getElementById('resultSummary');
    if (result.class === '健康') {
        summary.innerHTML = `
            <strong>检测结果良好！</strong><br>
            该柑橘样本被判定为<strong>健康</strong>的可能性为${healthyPercent}%。<br>
            建议：继续保持良好的种植管理，定期检查。
        `;
    } else {
        summary.innerHTML = `
            <strong>检测到异常！</strong><br>
            该柑橘样本被判定为<strong>患虫</strong>的可能性为${diseasedPercent}%。<br>
            建议：立即隔离并进一步检查，考虑采取防治措施。
        `;
    }

    // 显示结果容器
    resultContainer.style.display = 'block';
    
    // 滚动到结果区域
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ==================== 聊天功能 ====================
function initChatInput() {
    // 自动调整文本框高度
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });
}

function toggleSearchMode() {
    isSearching = !isSearching;
    isRAGMode = false;
    
    btnSearch.style.background = isSearching ? 'rgba(102, 126, 234, 0.3)' : '';
    btnRAG.style.background = '';
    
    showMessage(isSearching ? '已启用联网搜索模式' : '已关闭联网搜索', 'info');
}

function toggleRAGMode() {
    isRAGMode = !isRAGMode;
    isSearching = false;
    
    btnRAG.style.background = isRAGMode ? 'rgba(102, 126, 234, 0.3)' : '';
    btnSearch.style.background = '';
    
    showMessage(isRAGMode ? '已启用知识库检索模式' : '已关闭知识库检索', 'info');
}

async function clearConversation() {
    if (!confirm('确定要清除对话历史吗？')) {
        return;
    }

    try {
        const response = await fetch(API_BASE_URL + '/api/chat/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: conversationId })
        });

        const data = await response.json();
        if (data.success) {
            // 清空消息区，只保留欢迎消息
            chatMessages.innerHTML = `
                <div class="message assistant">
                    <div class="message-content">
                        您好！我是柑橘检测助手，可以帮您：
                        <br>• 解读检测结果
                        <br>• 回答柑橘实蝇相关问题
                        <br>• 提供防治建议
                        <br><br>请上传柑橘图像或向我提问吧！
                    </div>
                </div>
            `;
            
            // 生成新的对话 ID
            conversationId = 'default_' + Date.now();
            showMessage('对话历史已清除', 'success');
        }
    } catch (error) {
        console.error('清除失败:', error);
        showMessage('清除失败', 'error');
    }
}

function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) {
        return;
    }

    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    sendChatMessage(message, null);
}

async function sendChatMessage(message, imageResult = null) {
    // 添加用户消息
    addMessage(message, 'user');

    showLoading('思考中...');

    try {
        const mode = isSearching ? 'search' : (isRAGMode ? 'rag' : 'chat');
        
        const response = await fetch(API_BASE_URL + '/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                conversation_id: conversationId,
                mode: mode,
                image_result: imageResult
            })
        });

        const data = await response.json();
        
        if (data.success) {
            addMessage(data.response, 'assistant');
        } else {
            showMessage(data.message || '响应失败', 'error');
        }
    } catch (error) {
        console.error('发送失败:', error);
        showMessage('发送失败，请重试', 'error');
    } finally {
        hideLoading();
    }
}

function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `
        <div class="message-content">${content.replace(/\n/g, '<br>')}</div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ==================== 加载动画 ====================
function showLoading(text = '处理中...') {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

// ==================== 消息提示 ====================
function showMessage(message, type = 'info') {
    // 简单的 alert 实现，可以改进为更优雅的 toast
    const colors = {
        info: '#667eea',
        success: '#48bb78',
        error: '#f56565'
    };
    
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // 可以添加 toast 通知
    if (type === 'error') {
        alert(message);
    }
}

// ==================== 图像质量评估 ====================
async function assessImageQuality(file) {
    try {
        const formData = new FormData();
        formData.append('image', file);
        
        const response = await fetch(API_BASE_URL + '/api/image-quality/assess', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP 错误：${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayQualityAssessment(data.quality, data.suggestions);
        } else {
            console.warn('质量评估失败:', data.message);
        }
    } catch (error) {
        console.error('质量评估错误:', error);
        // 质量评估失败不影响主要功能
    }
}

function displayQualityAssessment(quality, suggestions) {
    // 显示质量评估面板
    qualityAssessment.style.display = 'block';
    
    // 总体评分
    const score = quality.overall_score;
    const level = quality.quality_level;
    
    let scoreColor;
    if (score >= 80) scoreColor = '#48bb78'; // 绿色
    else if (score >= 60) scoreColor = '#ed8936'; // 橙色
    else if (score >= 40) scoreColor = '#ecc94b'; // 黄色
    else scoreColor = '#f56565'; // 红色
    
    qualityScoreEl.innerHTML = `综合得分：<span style="color: ${scoreColor}">${score}</span>/100 (${level})`;
    
    // 各项指标
    sharpnessValueEl.textContent = `${quality.sharpness.level} (${quality.sharpness.normalized * 100}%)`;
    brightnessValueEl.textContent = `${quality.brightness.level} (${quality.brightness.value})`;
    contrastValueEl.textContent = `${quality.contrast.level} (${quality.contrast.normalized * 100}%)`;
    colorValueEl.textContent = `${quality.color_richness.level} (${quality.color_richness.normalized * 100}%)`;
    
    // 改进建议
    if (suggestions && suggestions.length > 0) {
        qualitySuggestions.style.display = 'block';
        suggestionsList.innerHTML = suggestions.map(s => 
            `<div style="margin-bottom: 8px;">
                <strong>${s.aspect}:</strong> ${s.issue}<br>
                <span style="color: var(--citrus-orange);">💡 ${s.suggestion}</span>
            </div>`
        ).join('');
    } else {
        qualitySuggestions.style.display = 'none';
    }
}
