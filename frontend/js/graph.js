function drawGraph(data, container) {
    container.innerHTML = '';
    
    const width = container.clientWidth || 800;
    const height = 600;
    const margin = 50; // Отступ от границ
    
    const svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto;");

    // Конфигурация сил
    const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links)
            .id(d => d.id)
            .distance(150)
            .strength(0.05))
        .force("charge", d3.forceManyBody()
            .strength(-150))
        .force("center", d3.forceCenter(width/2, height/2)
            .strength(0.1))
        .force("collision", d3.forceCollide()
            .radius(40))
        .force("bounds", () => { // Новая сила для границ
            data.nodes.forEach(node => {
                const pushStrength = 0.1;
                
                // Отталкивание от левой границы
                if(node.x < margin) {
                    node.vx += (margin - node.x) * pushStrength;
                }
                // Отталкивание от правой границы
                if(node.x > width - margin) {
                    node.vx += (width - margin - node.x) * pushStrength;
                }
                // Отталкивание от верхней границы
                if(node.y < margin) {
                    node.vy += (margin - node.y) * pushStrength;
                }
                // Отталкивание от нижней границы
                if(node.y > height - margin) {
                    node.vy += (height - margin - node.y) * pushStrength;
                }
            });
        })
        .alphaDecay(0.02)
        .velocityDecay(0.4);
    
    // Определяем цветовую шкалу для ребер
    const weightExtent = d3.extent(data.links, d => d.weight);
    const colorScale = d3.scaleLinear()
        .domain(weightExtent)
        .range(["#ddd", "#333"]);
    
    // Определяем шкалу толщины ребер
    const widthScale = d3.scaleLinear()
        .domain(weightExtent)
        .range([1, 5]);
    
    // Добавляем ребра
    const link = svg.append("g")
        .selectAll("line")
        .data(data.links)
        .join("line")
        .attr("stroke", d => colorScale(d.weight))
        .attr("stroke-width", d => widthScale(d.weight))
        .attr("stroke-opacity", 0.6);
    
    // Добавляем узлы
    const node = svg.append("g")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .selectAll("circle")
        .data(data.nodes)
        .join("circle")
        .attr("r", 10)
        .attr("fill", "#69b3a2")
        .call(drag(simulation));
    
    // Добавляем метки
    const label = svg.append("g")
        .selectAll("text")
        .data(data.nodes)
        .join("text")
        .text(d => d.name)
        .attr("font-family", "Arial")
        .attr("font-size", 12)
        .attr("dx", 15)
        .attr("dy", 4);
    
    // Обновление позиций при каждом тике симуляции
    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
        
        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
        
        label
            .attr("x", d => d.x)
            .attr("y", d => d.y);
    });
    
    // Добавляем SVG в контейнер
    container.appendChild(svg.node());
    
    // Функция для перетаскивания узлов
    function drag(simulation) {
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }
}
