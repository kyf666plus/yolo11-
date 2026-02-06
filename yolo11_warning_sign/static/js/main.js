// 全局变量
let currentDetectionResult = null;
let availableModels = [];
let currentModelType = 'door';

// DOM元素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const resultSection = document.getElementById('resultSection');
const originalImage = document.getElementById('originalImage');
const detectedImage = document.getElementById('detectedImage');
const loading = document.getElementById('loading');

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeUpload();
    loadModels();
    loadHistory();
});

// 加载可用模型
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();

        if (data.success) {
            availableModels = data.models;
            currentModelType = data.current_model;

            // 渲染模型按钮
            renderModelButtons(data.models);

            // 更新当前模型信息
            const currentModel = data.models.find(m => m.is_current);
            if (currentModel) {
                updateCurrentModelInfo(currentModel);
            }

            // 更新历史筛选器
            updateHistoryFilter(data.models);
        } else {
            showMessage('加载模型列表失败', 'error');
        }
    } catch (error) {
        console.error('加载模型列表失败:', error);
        showMessage('无法连接到服务器', 'error');
    }
}

// 渲染模型按钮
function renderModelButtons(models) {
    const modelButtons = document.getElementById('modelButtons');

    if (models.length === 0) {
        modelButtons.innerHTML = '<p class="no-models">没有可用的模型</p>';
        return;
    }

    modelButtons.innerHTML = models.map(model => `
        <button class="model-btn ${model.is_current ? 'active' : ''}"
                data-model-type="${model.type}"
                onclick="switchModel('${model.type}')">
            <span class="model-btn-icon">${getModelIcon(model.type)}</span>
            <span class="model-btn-text">${model.name}</span>
            <span class="model-btn-badge">${model.num_classes} 类</span>
        </button>
    `).join('');
}

// 获取模型图标

function getModelIcon(modelType) {
    const icons = {
        'door': '🚪',                    // 门状态检测
        'vest': '🦺',                    // 反光背心检测
        'safety_belt': '🔒',             // 安全带检测
        'face_mask': '😷',               // 防护面罩检测
        'goggles': '🥽',                 // 🆕 护目镜检测
        'fire_extinguisher': '🧯',       // 灭火器检测
        'warning_sign': '⚠️',            // 警示标志检测
        'tripod': '📐',                  // 三脚架检测
        'safety_barrier': '🚧'           // 安全防护栏检测
    };
    return icons[modelType] || '🎯';
}

// 更新当前模型信息
function updateCurrentModelInfo(model) {
    document.getElementById('currentModelName').textContent = model.name;
    document.getElementById('currentModelDesc').textContent =
        `${model.description} (支持 ${model.num_classes} 个类别: ${model.class_names.join(', ')})`;
}

// 切换模型
async function switchModel(modelType) {
    if (modelType === currentModelType) {
        return; // 已经是当前模型
    }

    try {
        const response = await fetch('/api/switch_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType })
        });

        const data = await response.json();

        if (data.success) {
            currentModelType = modelType;

            // 更新按钮状态
            document.querySelectorAll('.model-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.modelType === modelType) {
                    btn.classList.add('active');
                }
            });

            // 更新当前模型信息
            const model = availableModels.find(m => m.type === modelType);
            if (model) {
                updateCurrentModelInfo(model);
            }

            showMessage(data.message, 'success');

            // 如果有检测结果，清除它
            if (currentDetectionResult) {
                showMessage('模型已切换，请重新上传图片进行检测', 'info');
            }
        } else {
            showMessage(data.error, 'error');
        }
    } catch (error) {
        console.error('切换模型失败:', error);
        showMessage('切换模型失败', 'error');
    }
}

// 更新历史筛选器
function updateHistoryFilter(models) {
    const filter = document.getElementById('historyFilter');

    // 保留"全部"选项
    filter.innerHTML = '<option value="all">全部</option>';

    // 添加模型选项
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.type;
        option.textContent = model.name;
        filter.appendChild(option);
    });
}

// 初始化上传功能
function initializeUpload() {
    // 点击上传区域
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // 文件选择
    fileInput.addEventListener('change', handleFileSelect);

    // 拖拽上传
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
}

// 处理文件选择
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

// 处理拖拽悬停
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

// 处理拖拽离开
function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

// 处理文件拖拽
function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// 处理文件
function processFile(file) {
    // 验证文件类型
    if (!file.type.startsWith('image/')) {
        showMessage('请选择图片文件！', 'error');
        return;
    }

    // 验证文件大小 (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showMessage('图片文件不能超过16MB！', 'error');
        return;
    }

    // 显示原始图片
    const reader = new FileReader();
    reader.onload = function(e) {
        originalImage.src = e.target.result;
        resultSection.style.display = 'block';

        // 滚动到结果区域
        resultSection.scrollIntoView({ behavior: 'smooth' });
    };
    reader.readAsDataURL(file);

    // 上传并检测
    uploadAndDetect(file);
}

// 上传并检测
async function uploadAndDetect(file) {
    const formData = new FormData();
    formData.append('image', file);

    // 显示加载状态
    showLoading(true);
    detectedImage.style.display = 'none';

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            displayDetectionResult(result);
            addToHistory(result);
        } else {
            showMessage(result.error || '检测失败', 'error');
        }

    } catch (error) {
        console.error('Detection error:', error);
        showMessage('检测过程中发生错误，请重试', 'error');
    } finally {
        showLoading(false);
    }
}

// 显示检测结果
function displayDetectionResult(result) {
    currentDetectionResult = result;

    // 显示检测后的图片
    detectedImage.src = result.result_image_url + '?t=' + Date.now(); // 防止缓存
    detectedImage.style.display = 'block';

    // 更新统计信息
    document.getElementById('usedModel').textContent = result.model_name || '未知';
    document.getElementById('detectionCount').textContent = result.detection_count;
    document.getElementById('avgConfidence').textContent =
        result.avg_confidence ? (result.avg_confidence * 100).toFixed(1) + '%' : '0%';
    document.getElementById('processTime').textContent =
        result.process_time ? result.process_time.toFixed(2) + 's' : '-';
    document.getElementById('imageSize').textContent =
        `${result.image_width} × ${result.image_height}`;

    // 更新详细检测结果表格
    updateDetectionTable(result.detections);

    showMessage(`检测完成！使用 ${result.model_name} 检测到 ${result.detection_count} 个目标`, 'success');
}

// 更新检测结果表格
function updateDetectionTable(detections) {
    const tbody = document.getElementById('detectionTableBody');
    tbody.innerHTML = '';

    if (!detections || detections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">未检测到目标</td></tr>';
        return;
    }

    detections.forEach((detection, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>
                <span class="class-badge">${detection.class_name}</span>
            </td>
            <td>
                <span class="confidence-badge" style="background-color: ${getConfidenceColor(detection.confidence)}">
                    ${(detection.confidence * 100).toFixed(1)}%
                </span>
            </td>
            <td>${detection.bbox.x}, ${detection.bbox.y}, ${detection.bbox.w}, ${detection.bbox.h}</td>
        `;
        tbody.appendChild(row);
    });
}

// 根据置信度获取颜色
function getConfidenceColor(confidence) {
    if (confidence >= 0.8) return '#51cf66';
    if (confidence >= 0.6) return '#ffd43b';
    return '#ff6b6b';
}

// 显示/隐藏加载状态
function showLoading(show) {
    loading.style.display = show ? 'flex' : 'none';
}

// 显示消息
function showMessage(message, type = 'info') {
    // 移除现有消息
    const existingMessages = document.querySelectorAll('.message');
    existingMessages.forEach(msg => msg.remove());

    // 创建新消息
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.innerHTML = `
        <span>${message}</span>
        <button class="message-close" onclick="this.parentElement.remove()">×</button>
    `;

    // 插入到容器顶部
    const container = document.querySelector('.container');
    container.insertBefore(messageDiv, container.firstChild);

    // 3秒后自动移除
    setTimeout(() => {
        if (messageDiv.parentElement) {
            messageDiv.remove();
        }
    }, 3000);
}

// 下载结果图片
function downloadResult() {
    if (!currentDetectionResult) {
        showMessage('没有可下载的结果', 'error');
        return;
    }

    const link = document.createElement('a');
    link.href = currentDetectionResult.result_image_url;
    link.download = `detection_result_${currentDetectionResult.model_type}_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showMessage('结果图片已下载', 'success');
}

// 重新检测
function resetDetection() {
    resultSection.style.display = 'none';
    fileInput.value = '';
    currentDetectionResult = null;

    // 滚动到上传区域
    uploadArea.scrollIntoView({ behavior: 'smooth' });
}

// 添加到历史记录
function addToHistory(result) {
    let history = JSON.parse(localStorage.getItem('detectionHistory') || '[]');

    const historyItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleString('zh-CN'),
        model_type: result.model_type,
        model_name: result.model_name,
        detection_count: result.detection_count,
        avg_confidence: result.avg_confidence,
        process_time: result.process_time,
        image_size: `${result.image_width} × ${result.image_height}`,
        detections: result.detections
    };

    history.unshift(historyItem);

    // 只保留最近50条记录
    if (history.length > 50) {
        history = history.slice(0, 50);
    }

    localStorage.setItem('detectionHistory', JSON.stringify(history));
    loadHistory();
}

// 加载历史记录
function loadHistory() {
    const history = JSON.parse(localStorage.getItem('detectionHistory') || '[]');
    const historyList = document.getElementById('historyList');
    const filter = document.getElementById('historyFilter').value;

    // 筛选历史
    const filteredHistory = filter === 'all'
        ? history
        : history.filter(item => item.model_type === filter);

    if (filteredHistory.length === 0) {
        historyList.innerHTML = '<p class="no-history">暂无检测历史</p>';
        return;
    }

    historyList.innerHTML = filteredHistory.map(item => `
        <div class="history-item" onclick="showHistoryDetails(${item.id})">
            <div class="history-item-header">
                <span class="history-model-badge">${getModelIcon(item.model_type)} ${item.model_name}</span>
                <span class="history-time">${item.timestamp}</span>
            </div>
            <div class="history-item-content">
                <div class="history-stat">
                    <strong>${item.detection_count}</strong> 个目标
                </div>
                <div class="history-stat">
                    置信度: <strong>${item.avg_confidence ? (item.avg_confidence * 100).toFixed(1) + '%' : 'N/A'}</strong>
                </div>
                <div class="history-stat">
                    耗时: <strong>${item.process_time ? item.process_time.toFixed(2) + 's' : 'N/A'}</strong>
                </div>
                <div class="history-stat">
                    尺寸: <strong>${item.image_size}</strong>
                </div>
            </div>
        </div>
    `).join('');
}

// 筛选历史
function filterHistory() {
    loadHistory();
}

// 显示历史详情
function showHistoryDetails(id) {
    const history = JSON.parse(localStorage.getItem('detectionHistory') || '[]');
    const item = history.find(h => h.id === id);

    if (item) {
        let detailsHtml = `
            <strong>检测详情</strong><br><br>
            时间: ${item.timestamp}<br>
            模型: ${item.model_name}<br>
            检测数量: ${item.detection_count}<br>
            平均置信度: ${item.avg_confidence ? (item.avg_confidence * 100).toFixed(1) + '%' : 'N/A'}<br>
            处理时间: ${item.process_time ? item.process_time.toFixed(2) + 's' : 'N/A'}<br>
            图片尺寸: ${item.image_size}
        `;

        if (item.detections && item.detections.length > 0) {
            detailsHtml += '<br><br><strong>检测到的目标:</strong><br>';
            item.detections.forEach((det, idx) => {
                detailsHtml += `${idx + 1}. ${det.class_name} (${(det.confidence * 100).toFixed(1)}%)<br>`;
            });
        }

        alert(detailsHtml);
    }
}