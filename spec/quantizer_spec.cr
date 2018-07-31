require "./spec_helper"

describe Quantizer do
  # TODO: Write tests

  it "works" do
    arr = [] of Float32
    codes = (0...(3*512)).map { |_| 0_u8 }
    (0...512).each do |i|
      (0...5).each { |j| arr << 1_f32*(i/2) + j }
    end
    # 512 * 9
    q = Quantizer.new(5, 2)
    q.train(512, arr.to_unsafe, 100)
    q.assign_codes(arr.to_unsafe, codes.to_unsafe, 512)
    (0...256).each do |i|
      (0...3).each do |j|
        codes[i*2*3 + j].should eq(codes[(i*2 + 1)*3 + j])
        puts "mat[#{i*2}, #{j}], code: #{codes[i*2*3 + j]}, centroid: #{q.centroids(j, codes[i*2*3 + j]).value}, real: #{arr[i*2*5 + j*2]}"
        puts "mat[#{i*2 + 1}, #{j}], code: #{codes[(i*2 + 1)*3 + j]}, centroid: #{q.centroids(j, codes[(i*2 + 1)*3 + j]).value}, real: #{arr[(i*2 + 1)*5 + j*2]}"
      end
    end
    q.each_centroid(codes.to_unsafe) do |ix, val|
      puts "ix:#{ix}, val:#{val}"
    end
  end
end
