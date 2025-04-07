import{_ as a,c as o,a0 as t,o as c}from"./chunks/framework.OqxnQCTf.js";const b=JSON.parse('{"title":"Transformer","description":"","frontmatter":{"outline":[1,6]},"headers":[],"relativePath":"深度学习/李宏毅/2023年&2024年/5.Transformer.md","filePath":"深度学习/李宏毅/2023年&2024年/5.Transformer.md"}'),r={name:"深度学习/李宏毅/2023年&2024年/5.Transformer.md"};function s(d,e,l,n,i,h){return c(),o("div",null,e[0]||(e[0]=[t('<h1 id="transformer" tabindex="-1">Transformer <a class="header-anchor" href="#transformer" aria-label="Permalink to &quot;Transformer&quot;">​</a></h1><ol><li>文字转化为 token</li><li>理解 token：包括语意、位置、上下文 =&gt; 向量</li><li>Transform Block N：多个向量 =&gt;</li><li>Output</li></ol><h2 id="类神经网络技术使用历史" tabindex="-1">类神经网络技术使用历史 <a class="header-anchor" href="#类神经网络技术使用历史" aria-label="Permalink to &quot;类神经网络技术使用历史&quot;">​</a></h2><p><code>N-gram</code> -&gt; <code>Feed-forward Network</code> -&gt; <code>RNN</code> -&gt; <code>Transformer</code></p><h2 id="transformer核心步骤" tabindex="-1">Transformer核心步骤 <a class="header-anchor" href="#transformer核心步骤" aria-label="Permalink to &quot;Transformer核心步骤&quot;">​</a></h2><p><img src="https://github.com/user-attachments/assets/b836b98b-2c5b-4d89-9921-d63e97a85de1" alt="Image"></p><h3 id="根据输入内容生成token" tabindex="-1">根据输入内容生成token <a class="header-anchor" href="#根据输入内容生成token" aria-label="Permalink to &quot;根据输入内容生成token&quot;">​</a></h3><p>每一个大模型都有一个 <code>token list</code>，可以人工生成，也可以通过一定方式生成 <code>token list</code>，但是这些 <code>token list</code>都是大模型自己决定的，并不都相同</p><blockquote><p>可能中文中，每一个字就是一个 <code>token</code> ，在英语中，可能一个单词就是一个 <code>token</code> ，也有可能一个单词就是2个 <code>token</code>，数字也有可能一个单词就是2个 <code>token</code>，比如在 <code>GPT-3.5&amp;GPT-4</code>中，<code>1980</code>可能会被拆为 <code>198</code> 和 <code>0</code> 两个 token</p></blockquote><p><img src="https://github.com/user-attachments/assets/3b62a8b2-b210-4d06-ba7c-55819a157cf1" alt="Image"></p><h3 id="理解每一个token" tabindex="-1">理解每一个token <a class="header-anchor" href="#理解每一个token" aria-label="Permalink to &quot;理解每一个token&quot;">​</a></h3><h3 id="语意" tabindex="-1">语意 <a class="header-anchor" href="#语意" aria-label="Permalink to &quot;语意&quot;">​</a></h3><p>每一个 <code>token</code> 都可以变成 <code>向量</code>，<code>向量</code> 可以得到token之间的关联性和语意，为后面做准备</p><blockquote><p>本质就是通过一个训练好的模型得到对应的向量，但是向量是没考虑上下文的，也就是苹果（苹果手机还是苹果这种水果都是同一个向量）</p></blockquote><p><img src="https://github.com/user-attachments/assets/7f1095a2-c5fe-421f-948e-9a27f507e321" alt="Image"></p><h3 id="位置" tabindex="-1">位置 <a class="header-anchor" href="#位置" aria-label="Permalink to &quot;位置&quot;">​</a></h3><p>可以人工决定规则/训练得到一定的规则，相当于在原来加上语意的基础上 加上 位置的向量，为后面做准备</p><p><img src="https://github.com/user-attachments/assets/f99ceab2-49a0-4b20-9504-6680ae4545a4" alt="Image"></p><h3 id="上下文" tabindex="-1">上下文 <a class="header-anchor" href="#上下文" aria-label="Permalink to &quot;上下文&quot;">​</a></h3><p>根据一定的方式计算出 <code>当前token</code> 与 <code>其它token</code> 的相关性，然后加起来就是 <code>新的向量</code></p><blockquote><p>实际只会考虑 <code>当前token</code> 与 <code>前面的其它token</code> 的相关性</p></blockquote><p><img src="https://github.com/user-attachments/assets/b842bbf9-c946-4fa0-b6fc-e337fa9b46fb" alt="Image"></p><hr><p><img src="https://github.com/user-attachments/assets/2b2a2128-bb5b-4592-8789-bf0a45a08bcf" alt="Image"></p><hr><h3 id="transformer-block" tabindex="-1">Transformer Block <a class="header-anchor" href="#transformer-block" aria-label="Permalink to &quot;Transformer Block&quot;">​</a></h3><p>实际上不会只用一个 <code>Attention</code> _计算出相关性，因为可能有多种相关性！！这些向量是互相独立的</p><p><img src="https://github.com/user-attachments/assets/5dd1f7b2-c743-447c-aa13-8ba6502bfd69" alt="Image"></p><hr><p>最终多个向量需要合并为一个向量</p><p><img src="https://github.com/user-attachments/assets/85b9272b-2333-4408-9992-d6a10867e1bb" alt="Image"></p><p>上面这种流程就是 <code>Transformer Block</code></p><hr><p>但是实际是不止一个 <code>Transformer Block</code>的！</p><p><img src="https://github.com/user-attachments/assets/e8123814-bfc5-4703-947d-490a9b79ee93" alt="Image"></p><hr><h3 id="output" tabindex="-1">Output <a class="header-anchor" href="#output" aria-label="Permalink to &quot;Output&quot;">​</a></h3><p>经过多个 <code>Transformer Block</code> 的转化，最终我们得到一个输出！</p><blockquote><p>下图是一个简化过程，实际还不止下面的流程</p></blockquote><p><img src="https://github.com/user-attachments/assets/ea79342e-f234-4181-bed3-4ed966bf0946" alt="Image"></p><h3 id="其它细节总结" tabindex="-1">其它细节总结 <a class="header-anchor" href="#其它细节总结" aria-label="Permalink to &quot;其它细节总结&quot;">​</a></h3><blockquote><p>为什么我们只需要考虑 <code>当前token</code> 与 <code>前面的其它token</code> 的相关性</p></blockquote><p>因为答案是下图的一个流程，当我们生成 <code>w1</code> 时，下一次我们会将 <code>w1</code> 也作为输入去生成 <code>w2</code>，因此我们只需要考虑 <code>w1</code> 和 <code>它前面的token</code> 的相关性，因为我们后面的token都还没出来</p><blockquote><p>而对于 <code>w1</code> 前面的 token，需不需要计算跟 <code>w1</code> 的相关性呢？</p></blockquote><p>通过实践证明，<code>w1</code> 前面的 token，计不计算跟 <code>w1</code> 的相关性，其实效果都差不多，因此就直接不计算了！</p><p><img src="https://github.com/user-attachments/assets/c6d768b6-32cc-4e4d-b453-591189c94a1c" alt="Image"></p><hr><p>因此一个大模型的出现，总是会强调它能支持多长的 token，那是因为每一个 token 的增加，都需要更大算力的支持！</p><p><img src="https://github.com/user-attachments/assets/940e0898-0bcd-44e8-ae18-0ee2bd55b5b6" alt="Image"></p><h2 id="未来的研究方向-大模型每个流程的内容" tabindex="-1">未来的研究方向-大模型每个流程的内容 <a class="header-anchor" href="#未来的研究方向-大模型每个流程的内容" aria-label="Permalink to &quot;未来的研究方向-大模型每个流程的内容&quot;">​</a></h2><p><img src="https://github.com/user-attachments/assets/f4debcf5-1109-46a1-acec-9a9f4f17d198" alt="Image"></p><h3 id="分析每一个流程" tabindex="-1">分析每一个流程 <a class="header-anchor" href="#分析每一个流程" aria-label="Permalink to &quot;分析每一个流程&quot;">​</a></h3><ol><li>找出影响输出的输入：比如通过屏蔽输入的某一个字去验证对某一个输出的影响</li><li>找出影响输出的训练资料，到底是哪一篇文章或者多篇文章导致目前的输出内容（占比较大的训练资料）</li><li>分析<code>Embedding</code>有什么信息：比如有没有词性？每一个Transformer输出的<code>Embedding</code>到底是什么内容？</li></ol><blockquote><p>目的：当我们了解每一个流程蕴含的信息，我们就可以对这个模型进行底层的优化，加速推理/节省算力等等</p></blockquote><blockquote><p>依赖于将整个训练流程都开源的大模型信息 =&gt; 研究推出论文，不断深化理解复杂流程</p></blockquote><h3 id="直接问大模型" tabindex="-1">直接问大模型 <a class="header-anchor" href="#直接问大模型" aria-label="Permalink to &quot;直接问大模型&quot;">​</a></h3><ol><li>直接问影响输出的训练资料是什么</li><li>直接问得到的输出的信心概率有多少</li><li>直接问每一个输入影响输出的比重</li></ol>',57)]))}const p=a(r,[["render",s]]);export{b as __pageData,p as default};
