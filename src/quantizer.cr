require "./quantizer/*"

class Quantizer
  NBits               = 8
  KSub                = 1 << Quantizer::NBits
  MaxPointsPerCluster = 256
  MaxPoints           = MaxPointsPerCluster * KSub
  EPS                 = 1e-7

  @dim : Int32
  @dsub : Int32 # 将原来的向量划分为 nsubq 个 dsub 长度的向量
  @nsubq : Int32
  @dsub : Int32
  @centroids : Float32*
  @lastdsub : Int32 # 最后一个sub vector 可能是不完整的，所以单独存储长度

  def dist_l2(x : Float32*, y : Float32*, d : Int32)
    (0...d).reduce(0_f32) do |acc, i|
      tmp = x[i] - y[i]
      acc += tmp * tmp
    end
  end

  def initialize(@dim, @dsub, @centroids = Pointer(Float32).malloc(dim * KSub, 0_f32))
    @nsubq = @dim / @dsub
    @lastdsub = @dim % @dsub
    if @lastdsub == 0
      @lastdsub = @dsub
    else
      @nsubq += 1
    end
  end

  def centroids(m : Int32, i : UInt8) : Float32*
    if m == @nsubq - 1
      return @centroids + m * KSub * @dsub + i * @lastdsub
    end
    return @centroids + (m * KSub + i) * @dsub
  end

  def assign_centroid(x : Float32*, c0 : Float32*, code : UInt8*, d : Int32) : Float32
    # 给 x 的第 0 个 sub_vector 赋 cluster id
    c = c0
    dis = dist_l2 x, c0, d
    code.value = 0_u8
    (1...KSub).each do |j|
      c += d
      dis_ij = dist_l2(x, c, d)
      if dis_ij < dis
        code.value = j.to_u8
        dis = dis_ij
      end
    end
    return dis
  end

  def e_step(x : Float32*, centroids : Float32*, codes : UInt8*, n : Int32, d : Int32)
    # 赋予 cluster
    (0...n).each do |i|
      assign_centroid(x + i * d, centroids, codes + i, d)
    end
  end

  def m_step(x0 : Float32*, centroids : Float32*, codes : UInt8*, n : Int32, d : Int32)
    # 根据聚类，重新计算centroids
    nelts = StaticArray(Int32, KSub).new(0)
    centroids.clear d * KSub
    x = x0
    (0...n).each do |i|
      k = codes[i]
      c = centroids + k * d
      (0...d).each do |j|
        c[j] += x[j]
      end
      nelts[k] += 1
      x += d
    end
    c = centroids
    (0...KSub).each do |k|
      z = nelts[k]
      if z != 0
        (0...d).each do |j|
          c[j] /= z
        end
      end
      c += d
    end
    # 重新计算类中心完毕
    rand = Random.new
    # 给没有元素的类也赋一个中心,
    # 这个中心是随机找一个类，将类一分为2
    (0...KSub).each do |k|
      if nelts[k] == 0
        m = 0
        # 随机找一个centroid
        while rand.rand * (n - KSub) >= nelts[m] - 1
          m = (m + 1) % KSub
        end
        (centroids + k*d).copy_from (centroids + m * d), d
        (0...d).each do |j|
          sign = (j % 2) * 2 - 1
          centroids[k * d + j] += sign * EPS
          centroids[m * d + j] -= sign * EPS
        end
        nelts[k] = nelts[m] / 2
        nelts[m] -= nelts[k]
      end
    end
  end

  def kmeans(x : Float32*, c : Float32*, n : Int32, d : Int32, iter : Int32)
    perm = (0...n).to_a
    perm.shuffle!
    (0...KSub).each do |i|
      # 随机选一个x作为初始
      (c + i*d).copy_from(x + perm[i]*d, d)
    end
    codes = Pointer(UInt8).malloc(n, 0)
    (0...iter).each do |i|
      # STDERR.puts "iter:#{i}"
      e_step(x, c, codes, n, d)
      m_step(x, c, codes, n, d)
    end
  end

  def train(n : Int32, x : Float32*, iter : Int32 = 25)
    if n < KSub
      raise "Matrix too small for quantization, must have at least #{KSub} rows"
    end
    perm = (0...n).to_a
    d = @dsub
    np = n < MaxPoints ? n : MaxPoints
    xslice = Pointer(Float32).malloc(np * @dsub)
    (0...@nsubq).each do |m|
      d = @lastdsub if m == @nsubq - 1
      perm.shuffle! if np != n
      # 其实就是shuffle了一下x
      (0...np).each do |j|
        (xslice + j * d).copy_from(x + perm[j] * @dim + m * @dsub, d)
      end
      kmeans xslice, centroids(m, 0), np, d, iter
    end
  end

  def assign_code(x : Float32*, code : UInt8*)
    d = @dsub
    (0...@nsubq).each do |m|
      d = @lastdsub if m == @nsubq - 1
      assign_centroid (x + m * @dsub), centroids(m, 0), code + m, d
    end
  end

  def assign_codes(x : Float32*, code : UInt8*, n : Int32)
    (0...n).each do |i|
      assign_code x + i * @dim, code + i * @nsubq
    end
  end

  def each_centroid(codes : UInt8*)
    d = @dsub
    (0...@nsubq).each do |m|
      c = centroids m, codes[m]
      d = @lastdsub if m == @nsubq - 1
      (0...d).each do |n|
        yield m * @dsub + n, c[n]
      end
    end
  end

  def to_io(io : IO, format : IO::ByteFormat)
    @dim.to_io io, format
    @nsubq.to_io io, format
    @dsub.to_io io, format
    @lastdsub.to_io io, format
    (0...(@dim * KSub)).each do |i|
      @centroids[i].to_io io, format
    end
  end

  def self.from_io(io : IO, format : IO::ByteFormat)
    dim = Int32.from_io io, format
    nsubq = Int32.from_io io, format
    dsub = Int32.from_io io, format
    lastdsub = Int32.from_io io, format
    centroids = Pointer(Float32).malloc(dim * KSub, 0_f32)
    (0...(dim * KSub)).each do |i|
      @centroids[i] = Float32.from_io io, format
    end
    return Quantizer.new(dim, dsub, centroids)
  end
end
